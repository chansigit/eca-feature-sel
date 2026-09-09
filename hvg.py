"""Streaming HVG scores per group (dataset-level + coarse lineages), two methods.

Never materializes the full matrix; `layers/counts` is scanned in bounded blocks.

Groups: 0 = every kept cell (the dataset-level call), 1..k = lineages that pass
the cell-count filter. Per cell, `grp` holds the lineage code, -1 for "kept but
in no lineage", -2 for "dropped cell" (numerical guard: too few counts).

vst (Seurat v3 / scanpy seurat_v3)
    1. per gene mean/variance of raw counts
    2. loess log10(var) ~ log10(mean) -> expected sd (quadratic fit as fallback)
    3. score = sum_cells clip((x-mean)/sd, sqrt(N))^2 / (N-1)
    Zero entries have the closed form (N - n_detected)*(mean/sd)^2, so pass 2 only
    touches stored nonzeros.

pearson (analytic Pearson residuals, Lause/Berens/Kobak 2021 / scanpy experimental)
    mu_ij = depth_i * p_j,  var_ij = mu + mu^2/theta,  z = clip((x-mu)/sqrt(var), +-sqrt(N))
    score = Var_cells(z)  (1 = pure NB noise at theta)
    mu depends on the cell, so there is no zero closed form: pass 2 walks dense
    row blocks (a few hundred cells x all genes) -- O(N*G) but vectorized.

Both scores have a natural reference (1 = "as expected"), which is what makes
them comparable across groups and datasets; ranks are not.
"""
import numpy as np

CHUNK = 20_000_000    # stored entries per sparse read block
DENSE_ELEMS = 5_000_000  # elements per dense row block (rows = this // n_vars)


# ------------------------------------------------------------------ block readers
def _sparse_chunks(g, enc):
    """Yield (gene, cell, value) for stored entries, in bounded blocks."""
    indptr = np.asarray(g["indptr"][:])
    indices, data = g["indices"], g["data"]
    nnz = int(data.shape[0])
    for s in range(0, nnz, CHUNK):
        e = min(s + CHUNK, nnz)
        idx = np.asarray(indices[s:e])
        val = np.asarray(data[s:e], dtype=np.float64)
        pos = np.searchsorted(indptr, np.arange(s, e), side="right") - 1
        yield (idx, pos, val) if enc == "csr_matrix" else (pos, idx, val)


def _dense_row_blocks(g, enc, n_obs, n_vars):
    """Yield (row_start, dense block) -- for the per-cell-mu methods."""
    rows = max(1, DENSE_ELEMS // max(n_vars, 1))
    if enc == "csr_matrix":
        indptr = np.asarray(g["indptr"][:])
        for s in range(0, n_obs, rows):
            e = min(s + rows, n_obs)
            a, b = int(indptr[s]), int(indptr[e])
            block = np.zeros((e - s, n_vars))
            if b > a:
                r = np.repeat(np.arange(e - s), np.diff(indptr[s:e + 1]))
                block[r, np.asarray(g["indices"][a:b])] = np.asarray(g["data"][a:b], dtype=np.float64)
            yield s, block
    elif enc == "csc_matrix":
        # csc has no cheap row slices; load once and convert (rare: none in corpus)
        from scipy import sparse
        X = sparse.csc_matrix((np.asarray(g["data"][:]), np.asarray(g["indices"][:]),
                               np.asarray(g["indptr"][:])), shape=(n_obs, n_vars)).tocsr()
        for s in range(0, n_obs, rows):
            yield s, X[s:s + rows].toarray().astype(np.float64)
    else:
        for s in range(0, g.shape[0], rows):
            yield s, np.asarray(g[s:s + rows], dtype=np.float64)


class Reader:
    """One h5 matrix, read from disk once: the first pass over sparse chunks or
    dense blocks is cached in RAM (up to `budget` bytes) and replayed by the
    later passes. Keeps the streaming code identical whether cached or not."""

    def __init__(self, g, enc, n_obs, n_vars, budget=6e9):
        self.g, self.enc, self.n_obs, self.n_vars, self.budget = g, enc, n_obs, n_vars, budget
        # scRNA matrices stored dense are still ~97% zeros: the stored-entry path is
        # far cheaper, so every encoding is served as sparse chunks (dense_blocks stays
        # for the per-cell-mu methods).
        self.sparse = True
        self.stored_sparse = enc in ("csr_matrix", "csc_matrix")
        self._sparse_cache = self._dense_cache = None

    def _replay(self, attr, gen, nbytes):
        cached = getattr(self, attr)
        if cached is not None:
            yield from cached
            return
        keep, total = [], 0
        for item in gen:
            if keep is not None:
                total += nbytes(item)
                keep = keep + [item] if total <= self.budget else None
            yield item
        setattr(self, attr, keep)

    def sparse_chunks(self):
        gen = _sparse_chunks(self.g, self.enc) if self.stored_sparse else self._dense_as_sparse()
        return self._replay("_sparse_cache", gen, lambda c: sum(a.nbytes for a in c))

    def _dense_as_sparse(self):
        for s, block in self.dense_blocks():
            cell, gene = np.nonzero(block)
            yield gene, cell + s, block[cell, gene]

    def dense_blocks(self):
        return self._replay("_dense_cache", _dense_row_blocks(self.g, self.enc, self.n_obs, self.n_vars),
                            lambda b: b[1].nbytes)


# ------------------------------------------------------------------ cell guard
def cell_depth(src):
    """Total counts per cell (also primes the Reader cache)."""
    depth = np.zeros(src.n_obs)
    if src.sparse:
        for gene, cell, val in src.sparse_chunks():
            depth += np.bincount(cell, weights=val, minlength=src.n_obs)
    else:
        for s, block in src.dense_blocks():
            depth[s:s + block.shape[0]] = block.sum(axis=1)
    return depth


# --------------------------------------------------------------------- pass 1
def _onehot(codes, n_grp):
    """(n_grp-1) x rows lineage indicator, for dense BLAS group sums."""
    m = np.zeros((n_grp - 1, len(codes)))
    ok = codes >= 0
    m[codes[ok], np.flatnonzero(ok)] = 1.0
    return m


def moments(src, grp, n_grp):
    """Pass 1 -> sums, sumsq, n_detected (each (n_grp, n_vars)) and per-cell depth.
    Row 0 = all kept cells. Dropped cells (grp == -2) are ignored everywhere."""
    n_obs, n_vars = src.n_obs, src.n_vars
    sums = np.zeros((n_grp, n_vars))
    sqs = np.zeros((n_grp, n_vars))
    ndet = np.zeros((n_grp, n_vars), dtype=np.int64)
    depth = np.zeros(n_obs)
    if src.sparse:
        # buckets 0..k-1 = lineages, bucket k = kept cells in no lineage; the
        # dataset-level row is the sum, so each chunk costs one bincount per stat.
        nb = n_grp  # (n_grp-1) lineages + 1 rest bucket
        bsum = np.zeros((nb, n_vars)); bsq = np.zeros((nb, n_vars)); bdet = np.zeros((nb, n_vars), dtype=np.int64)
        for gene, cell, val in src.sparse_chunks():
            depth += np.bincount(cell, weights=val, minlength=n_obs)
            lin = grp[cell]
            kept = lin >= -1
            gene, val, lin = gene[kept], val[kept], lin[kept]
            key = np.where(lin >= 0, lin, nb - 1) * n_vars + gene
            bsum += np.bincount(key, weights=val, minlength=nb * n_vars).reshape(nb, n_vars)
            bsq += np.bincount(key, weights=val * val, minlength=nb * n_vars).reshape(nb, n_vars)
            bdet += np.bincount(key[val != 0], minlength=nb * n_vars).reshape(nb, n_vars)
        sums[0], sqs[0], ndet[0] = bsum.sum(axis=0), bsq.sum(axis=0), bdet.sum(axis=0)
        if n_grp > 1:
            sums[1:], sqs[1:], ndet[1:] = bsum[:-1], bsq[:-1], bdet[:-1]
    else:
        for s, block in src.dense_blocks():
            depth[s:s + block.shape[0]] = block.sum(axis=1)
            codes = grp[s:s + block.shape[0]]
            block = block[codes >= -1]
            codes = codes[codes >= -1]
            nz = (block != 0).astype(np.float64)
            sums[0] += block.sum(axis=0)
            sqs[0] += (block * block).sum(axis=0)
            ndet[0] += nz.sum(axis=0).astype(np.int64)
            if n_grp > 1:
                oh = _onehot(codes, n_grp)
                sums[1:] += oh @ block
                sqs[1:] += oh @ (block * block)
                ndet[1:] += (oh @ nz).astype(np.int64)
    return sums, sqs, ndet, depth


# ------------------------------------------------------------------------ vst
def expected_sd(mean, var, valid, span, notes):
    """loess of log10(var) ~ log10(mean) over `valid` genes; quadratic fallback.
    skmisc raises on small/sparse groups (singular design), which would otherwise
    cost the whole lineage."""
    sd = np.zeros_like(mean)
    ok = valid & (mean > 0) & (var > 0)
    if ok.sum() < 20:
        notes.append("too few expressed genes to fit")
        return sd, ok
    x, y = np.log10(mean[ok]), np.log10(var[ok])
    fitted = None
    try:
        from skmisc.loess import loess
        model = loess(x, y, span=span, degree=2)
        model.fit()
        fitted = np.asarray(model.outputs.fitted_values)
        if not np.all(np.isfinite(fitted)):
            fitted = None
            notes.append("loess returned non-finite values, used quadratic fit")
    except Exception as e:
        notes.append(f"loess failed ({type(e).__name__}), used quadratic fit")
    if fitted is None:
        fitted = np.polyval(np.polyfit(x, y, 2), x)
    sd[ok] = np.sqrt(10.0 ** fitted)
    return sd, ok & (sd > 0)


def vst_score(src, grp, n_grp, sizes, mean, sd, ndet, valid):
    """Pass 2 -> clipped standardized variance, (n_grp, n_vars)."""
    n_vars = src.n_vars
    safe = np.where(sd > 0, sd, 1.0)
    clip = np.sqrt(sizes)
    acc = np.zeros((n_grp, n_vars))
    if src.sparse:
        mean_f, safe_f = mean.ravel(), safe.ravel()
        for gene, cell, val in src.sparse_chunks():
            lin = grp[cell]
            keep = (val != 0) & (lin >= -1)
            gene, val, lin = gene[keep], val[keep], lin[keep]
            z = np.minimum((val - mean_f[gene]) / safe_f[gene], clip[0])
            acc[0] += np.bincount(gene, weights=z * z, minlength=n_vars)
            if n_grp > 1:
                m = lin >= 0
                key = (lin[m] + 1) * n_vars + gene[m]
                z = np.minimum((val[m] - mean_f[key]) / safe_f[key], clip[lin[m] + 1])
                acc[1:] += np.bincount(key - n_vars, weights=z * z,
                                       minlength=(n_grp - 1) * n_vars).reshape(-1, n_vars)
    else:
        for s, block in src.dense_blocks():
            codes = grp[s:s + block.shape[0]]
            for k in range(n_grp):
                sub = block[codes >= -1] if k == 0 else block[codes == k - 1]
                if not len(sub):
                    continue
                z = np.minimum((sub - mean[k]) / safe[k], clip[k])
                acc[k] += np.where(sub != 0, z * z, 0.0).sum(axis=0)
    zeros = (sizes[:, None] - ndet) * (mean / safe) ** 2  # every zero cell: (-mean/sd)^2
    out = (acc + zeros) / np.maximum(sizes[:, None] - 1, 1)
    return np.where(valid, out, 0.0)


# ------------------------------------------------------------------- pearson
def pearson_score(src, grp, n_grp, sizes, sums, depth, valid, theta):
    """Pass 2 -> variance of clipped analytic Pearson residuals, (n_grp, n_vars).
    p_j is fitted within each group, so a lineage call asks about variance
    *inside* that lineage, not its identity markers."""
    n_vars = src.n_vars
    tot = np.array([depth[grp >= -1].sum()] + [depth[grp == k].sum() for k in range(n_grp - 1)])
    p = sums / np.where(tot > 0, tot, 1.0)[:, None]
    clip = np.sqrt(sizes)
    s1 = np.zeros((n_grp, n_vars))
    s2 = np.zeros((n_grp, n_vars))
    for s, block in src.dense_blocks():
        codes = grp[s:s + block.shape[0]]
        d = depth[s:s + block.shape[0]]
        for k in range(n_grp):
            rows = codes >= -1 if k == 0 else codes == k - 1
            if not rows.any():
                continue
            mu = d[rows, None] * p[k][None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (block[rows] - mu) / np.sqrt(mu + mu * mu / theta)
            np.nan_to_num(z, copy=False)
            np.clip(z, -clip[k], clip[k], out=z)
            s1[k] += z.sum(axis=0)
            s2[k] += (z * z).sum(axis=0)
    n = np.maximum(sizes, 1)[:, None]
    var = s2 / n - (s1 / n) ** 2
    return np.where(valid, var, 0.0)


# ---------------------------------------------------------------------- driver
def call(src, grp, sizes, methods=("vst",), span=0.3, theta=100.0, min_gene_cells=3, notes=None):
    """Scores for each method and group. -> dict method -> (n_grp, n_vars) array,
    plus (ndet_all, sums_all, mean_all) for the expression stats.

    `sizes[0]` must be the number of kept cells (grp >= -1)."""
    n_grp = len(sizes)
    notes = {} if notes is None else notes
    sums, sqs, ndet, depth = moments(src, grp, n_grp)
    N = np.maximum(sizes, 1)[:, None].astype(float)
    mean = sums / N
    var = (sqs - N * mean ** 2) / np.maximum(N - 1, 1)
    # numerical guard: a gene seen in fewer cells than this is not rankable here
    valid = (ndet >= min_gene_cells) & (mean > 0) & (var > 0)
    out = {}
    if "vst" in methods:
        sd = np.zeros_like(mean)
        ok = np.zeros(mean.shape, dtype=bool)
        for i in range(n_grp):
            n = []
            sd[i], ok[i] = expected_sd(mean[i], var[i], valid[i], span, n)
            if n:
                notes.setdefault(i, []).extend(n)
        out["vst"] = vst_score(src, grp, n_grp, sizes, mean, sd, ndet, ok)
        out["vst_trend"] = sd ** 2
    if "pearson" in methods:
        out["pearson"] = pearson_score(src, grp, n_grp, sizes, sums, depth, valid, theta)
        tot = np.array([depth[grp >= -1].sum()] + [depth[grp == k].sum() for k in range(n_grp - 1)])
        mu_bar = sums / np.where(tot > 0, tot, 1.0)[:, None] * (tot / np.maximum(sizes, 1))[:, None]
        out["pearson_trend"] = mu_bar + mu_bar ** 2 / theta  # expected var at mean depth
    return out, ndet, sums, mean, var, valid


def ranks(score, valid):
    """1-based rank by descending score within each group; 0 = not rankable."""
    r = np.zeros(score.shape, dtype=np.int32)
    for i in range(score.shape[0]):
        order = np.argsort(-np.where(valid[i], score[i], -np.inf), kind="stable")
        rr = np.empty(len(order), dtype=np.int32)
        rr[order] = np.arange(1, len(order) + 1)
        r[i] = np.where(valid[i], rr, 0)
    return r


def top_n(score, valid, n):
    """Boolean (n_grp, n_vars): the n highest-scoring rankable genes per group."""
    r = ranks(score, valid)
    return (r > 0) & (r <= n)
