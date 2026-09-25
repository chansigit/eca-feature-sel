#!/usr/bin/env python
"""Per-dataset measurement: one h5ad in, two parquets out.

Per harmonized gene (``var.gene_id_harmonized``), dataset-level (all kept cells):

    n_cells, n_detected, sum_counts, var, expected_var, score, rank, hvg_upstream

plus a ``.lineages.parquet`` with (lineage, lineage_cells, gene, score, rank) for
the top `n_store` genes of every kept lineage. Selection itself (s_min, n_max,
lineage size, how many lineages) happens at build time from these; nothing here
decides what is an HVG.

Nothing here builds an AnnData: `var`/`obs` are read as arrays and
``layers/counts`` is scanned twice with h5py (see hvg.py). ``var.highly_variable``
as the upstream pipeline left it is carried along for comparison.

    worker.py --manifest jobs/manifest.tsv --index $SLURM_ARRAY_TASK_ID
    worker.py --sample-key K --species human --h5ad X.h5ad --out out.parquet
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import pandas as pd

import featuresel
import hvg

CHUNK = 20_000_000  # nnz per read block; caps memory on the largest files
ROWS = 20_000       # row block for dense layers
MISSING_IDS = {"", "nan", "na", "none", "null"}
MISSING_LABELS = MISSING_IDS | {"unknown", "unassigned", "<NA>".lower()}


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr], dtype=object)


def _attr(node, name, default=""):
    v = node.attrs.get(name, default)
    return v.decode() if isinstance(v, bytes) else v


def _column(f, name, n_vars):
    """Read one var column, materializing an AnnData categorical if needed."""
    node = f.get(f"var/{name}")
    if node is None:
        return None
    if isinstance(node, h5py.Group):  # categories + codes
        cats = node["categories"][:]
        codes = np.asarray(node["codes"][:])
        out = np.empty(n_vars, dtype=object)
        out[:] = None
        ok = codes >= 0
        out[ok] = _decode(cats)[codes[ok]]
        return out
    arr = np.asarray(node[:])
    return _decode(arr) if arr.dtype.kind in "SO" else arr  # h5py hands back bytes


def harmonized_map(f, n_vars):
    """(keep mask over var rows, group code per kept row, unique gene ids)."""
    raw = _column(f, "gene_id_harmonized", n_vars)
    if raw is None:
        raise SystemExit("var.gene_id_harmonized not found")
    ids = np.array([None if x is None or (isinstance(x, float) and np.isnan(x))
                    or str(x).strip().lower() in MISSING_IDS else str(x).strip()
                    for x in raw], dtype=object)
    keep = np.array([x is not None for x in ids])
    if not keep.any():
        raise SystemExit("no valid harmonized gene ids")
    codes, uniq = pd.factorize(pd.Series(ids[keep]))
    return keep, np.asarray(codes), np.asarray(uniq, dtype=object)




def lineage_codes(f, n_obs, rule, meta, dropped=None):
    """cell -> kept-lineage code (-1 = none, -2 = dropped cell), plus kept lineage
    names/sizes.

    The obs key is the first of `lineage_obs_key` that exists. Lineages under
    `min_lineage_cells` (counted after dropping cells) are not made groups; their
    cells still count towards the dataset-level call.
    """
    keys = rule.get("lineage_obs_key") or []
    keys = [keys] if isinstance(keys, str) else keys
    cols = set(f["obs"].attrs.get("column-order", []).tolist()) if "obs" in f else set()
    key = next((k for k in keys if k in cols), None)
    meta["lineage_obs_key"] = key
    dropped = np.zeros(n_obs, dtype=bool) if dropped is None else dropped
    if key is None:
        return np.where(dropped, -2, -1).astype(np.int64), [], np.array([], dtype=np.int64)
    node = f[f"obs/{key}"]
    if isinstance(node, h5py.Group):
        cats = _decode(node["categories"][:])
        raw = np.asarray(node["codes"][:])
        labels = np.array([cats[c] if c >= 0 else None for c in raw], dtype=object)
    else:
        labels = _decode(np.asarray(node[:]))
    ok = np.array([x is not None and str(x).strip().lower() not in MISSING_LABELS
                   for x in labels]) & ~dropped
    names, sizes, codes = [], [], np.where(dropped, -2, -1).astype(np.int64)
    min_cells = int(rule.get("min_lineage_cells", 200))
    for name in pd.unique(labels[ok]):
        idx = np.flatnonzero(ok & (labels == name))
        if len(idx) < min_cells:
            continue
        codes[idx] = len(names)
        names.append(str(name))
        sizes.append(len(idx))
    meta["lineages_total"] = int(len(pd.unique(labels[ok])))
    meta["lineages_kept"] = len(names)
    return codes, names, np.array(sizes, dtype=np.int64)




def compute(cfg, h5ad, species, key, out):
    rule = cfg["hvg"]
    layer = rule.get("layer", "counts")
    meta = {"sample_key": key, "species": species, "h5ad": h5ad,
            "h5ad_mtime": os.path.getmtime(h5ad),
            "hvg_signature": featuresel.hvg_signature(cfg)}
    with h5py.File(h5ad, "r") as f:
        if f"layers/{layer}" not in f:
            raise SystemExit(f"{key}: layers/{layer} not found")
        g = f[f"layers/{layer}"]
        enc = _attr(g, "encoding-type", "array")
        shape = tuple(int(x) for x in (g.attrs["shape"] if "shape" in g.attrs else g.shape))
        n_obs, n_vars = shape
        keep, codes, ids = harmonized_map(f, n_vars)
        meta.update(n_cells=n_obs, n_vars_original=n_vars, encoding=enc,
                    n_unmapped_rows=int((~keep).sum()), n_genes_harmonized=len(ids),
                    n_duplicate_harmonized_ids=int(keep.sum() - len(ids)))

        # numerical guard, not QC: cells with (almost) no counts break both methods
        src = hvg.Reader(g, enc, n_obs, n_vars, budget=float(rule.get("cache_bytes", 6e9)))
        depth = hvg.cell_depth(src)
        dropped = depth < float(rule.get("min_cell_counts", 10))
        meta["n_cells_dropped"] = int(dropped.sum())
        lin_codes, lin_names, lin_sizes = lineage_codes(f, n_obs, rule, meta, dropped)
        sizes = np.concatenate([[int((~dropped).sum())], lin_sizes]).astype(np.int64)
        notes = {}
        method = rule.get("method", "vst")
        n_store = int(rule.get("n_store", 5000))
        t0 = time.perf_counter()
        if rule.get("compute", True):
            res, det_rows, sum_rows, mean_rows, var_rows, valid = hvg.call(
                src, lin_codes, sizes, methods=(method,),
                span=float(rule.get("span", 0.3)), theta=float(rule.get("theta", 100)),
                min_gene_cells=int(rule.get("min_gene_cells", 3)), notes=notes)
            score_rows = res[method]
            trend_rows = res[f"{method}_trend"]
            rank_rows = hvg.ranks(score_rows, valid)
        else:
            sum_rows, sq_rows, det_rows, _ = hvg.moments(src, lin_codes, len(sizes))
            N = np.maximum(sizes, 1)[:, None].astype(float)
            mean_rows = sum_rows / N
            var_rows = (sq_rows - N * mean_rows ** 2) / np.maximum(N - 1, 1)
            score_rows = trend_rows = np.full(mean_rows.shape, np.nan)
            rank_rows = np.zeros(mean_rows.shape, dtype=np.int32)
        meta["hvg_seconds"] = round(time.perf_counter() - t0, 2)
        meta["fit_notes"] = {(["dataset"] + lin_names)[i]: v for i, v in notes.items()}
        n_grp = len(sizes)

        # var rows -> harmonized genes. Sums add; detection is an upper bound for
        # the few duplicated ids (ponytail: exact would need another pass); score
        # takes the best row, rank the best (smallest nonzero) row.
        n_det = np.minimum(np.bincount(codes, weights=det_rows[0][keep],
                                       minlength=len(ids)).astype(np.int64), int(sizes[0]))
        sums = np.bincount(codes, weights=sum_rows[0][keep], minlength=len(ids))
        score = np.full((n_grp, len(ids)), np.nan)
        rank = np.zeros((n_grp, len(ids)), dtype=np.int32)
        var_g = np.zeros((n_grp, len(ids)))
        trend_g = np.full((n_grp, len(ids)), np.nan)
        for i in range(n_grp):
            score[i] = _best(codes, score_rows[i][keep], len(ids), np.fmax, np.nan)
            r = np.where(rank_rows[i][keep] > 0, rank_rows[i][keep], np.iinfo(np.int32).max)
            r = _best(codes, r, len(ids), np.minimum, np.iinfo(np.int32).max)
            rank[i] = np.where(r == np.iinfo(np.int32).max, 0, r)
            var_g[i] = _best(codes, var_rows[i][keep], len(ids), np.fmax, 0.0)
            trend_g[i] = _best(codes, trend_rows[i][keep], len(ids), np.fmax, np.nan)

        up = _column(f, "highly_variable", n_vars)
        if up is None:
            hvg_up = np.zeros(len(ids), dtype=bool)
            meta["upstream_hvg"] = None
        else:
            hvg_up = np.zeros(len(ids), dtype=bool)
            np.logical_or.at(hvg_up, codes, np.asarray(up, dtype=bool)[keep])
            uns_hvg = f.get("uns/hvg")
            flavor = uns_hvg["flavor"][()] if uns_hvg is not None and "flavor" in uns_hvg else ""
            meta["upstream_hvg"] = {
                "n": int(hvg_up.sum()),
                "flavor": flavor.decode() if isinstance(flavor, bytes) else str(flavor),
                "batch_aware": "highly_variable_nbatches" in f["var"]}
        sym = _column(f, "gene_symbol_harmonized", n_vars)
        symbol = (_best_str(codes, sym[keep], len(ids)) if sym is not None else ids)

    meta.update(method=method, n_store=n_store, groups=["dataset"] + lin_names,
                group_sizes=[int(x) for x in sizes],
                n_rankable=[int((rank[i] > 0).sum()) for i in range(n_grp)],
                score_at_rank={str(k): [float(np.nanmax(np.where(rank[i] == k, score[i], np.nan)))
                                        if (rank[i] == k).any() else None for i in range(n_grp)]
                               for k in (100, 500, 1000, 2000, 3000)})

    df = pd.DataFrame({"sample_key": key, "species": species, "harmonized_id": ids, "symbol": symbol,
                       "n_cells": int(sizes[0]), "n_detected": n_det, "sum_counts": sums,
                       "var": var_g[0], "expected_var": trend_g[0],
                       "score": score[0], "rank": rank[0], "hvg_upstream": hvg_up})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)  # atomic: a half-written parquet never looks "done"
    # per-(lineage, gene) detail for the top n_store genes of each lineage, so the
    # lineage-size / s_min / n_max thresholds can all move at build time
    rows = []
    for i, (name, size) in enumerate(zip(lin_names, lin_sizes), start=1):
        sel = (rank[i] > 0) & (rank[i] <= n_store)
        rows.append(pd.DataFrame({"sample_key": key, "lineage": name, "lineage_cells": int(size),
                                  "harmonized_id": ids[sel], "score": score[i][sel], "rank": rank[i][sel]}))
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["sample_key", "lineage", "lineage_cells", "harmonized_id", "score", "rank"])
    lin_out = os.path.splitext(out)[0] + ".lineages.parquet"
    detail.to_parquet(lin_out + ".tmp", index=False)
    os.replace(lin_out + ".tmp", lin_out)
    with open(os.path.splitext(out)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    s15 = int(np.nansum(score[0] >= 1.5))
    print(f"[{key}] {species} cells={sizes[0]}/{n_obs} genes={len(ids)} enc={enc} "
          f"lineages={len(lin_names)}/{meta.get('lineages_total', 0)} "
          f"score>=1.5: {s15} upstream={int(hvg_up.sum())} {meta['hvg_seconds']}s -> {out}", flush=True)


def _best(codes, values, n, op, fill):
    """Aggregate var-row values onto genes with a ufunc (fmax / minimum)."""
    out = np.full(n, fill, dtype=float)
    op.at(out, codes, np.asarray(values, dtype=float))
    return out


def _best_str(codes, values, n):
    """First non-empty symbol per gene."""
    out = np.array([""] * n, dtype=object)
    for c, v in zip(codes, values):
        if not out[c] and v is not None and str(v) not in ("", "nan"):
            out[c] = str(v)
    return out


def _run(args):
    cfg, line = args
    key, species, h5ad, out = line.split("\t")
    compute(cfg, h5ad, species, key, out)
    return key


def main():
    ap = argparse.ArgumentParser(description="eca-feature-sel per-dataset worker")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "config.yaml"))
    ap.add_argument("--manifest")
    ap.add_argument("--index", type=int, help="1-based line in manifest; omit to run all lines")
    ap.add_argument("--jobs", type=int, default=1, help="processes when running a whole manifest")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds a dataset may run before it is abandoned (Oak can hang a read)")
    ap.add_argument("--h5ad")
    ap.add_argument("--species")
    ap.add_argument("--sample-key")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = featuresel.load_config(a.config)
    if a.manifest:
        lines = [ln for ln in open(a.manifest).read().splitlines() if ln.strip()]
        lines = [lines[a.index - 1]] if a.index else lines
    else:
        lines = ["\t".join([a.sample_key, a.species, a.h5ad, a.out])]
    # biggest files first so the pool does not end on one 6 GB straggler; a file
    # whose size cannot even be read in time is deferred right away
    sizes, slow = featuresel.fs_parallel(os.path.getsize, [ln.split("\t")[2] for ln in lines])
    for ln in lines:
        if ln.split("\t")[2] in slow:
            print(f"[{ln.split(chr(9))[0]}] deferred: stat() did not return in {featuresel.FS_TIMEOUT}s")
    lines = sorted((ln for ln in lines if ln.split("\t")[2] in sizes),
                   key=lambda ln: -sizes[ln.split("\t")[2]])
    deferred = run_pool(cfg, lines, a.jobs, a.timeout)
    if deferred:
        print(f"{len(deferred)} dataset(s) deferred (no result within {a.timeout}s): {', '.join(deferred)}")
    featuresel.fs_exit(0)


def run_pool(cfg, lines, jobs, timeout):
    """One forked process per dataset, at most `jobs` at a time. A dataset still
    running after `timeout` s is killed and reported; nothing is written for it,
    so the next measure picks it up again. Hung children are never joined."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    pending, running, deferred = list(lines), {}, []
    while pending or running:
        while pending and len(running) < jobs:
            ln = pending.pop(0)
            p = ctx.Process(target=_run, args=((cfg, ln),), daemon=True)
            p.start()
            running[p] = (ln, time.time())
        time.sleep(0.5)
        for p in list(running):
            ln, t0 = running[p]
            key = ln.split("\t")[0]
            if not p.is_alive():
                p.join()
                del running[p]
                if p.exitcode:
                    print(f"[{key}] failed (exit {p.exitcode})", flush=True)
            elif time.time() - t0 > timeout:
                p.kill()
                del running[p]
                deferred.append(key)
                print(f"[{key}] deferred: no result after {timeout}s", flush=True)
    return deferred


if __name__ == "__main__":
    main()
