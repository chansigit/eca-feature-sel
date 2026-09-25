#!/usr/bin/env python
"""Self-check: python test_worker.py  (needs anndata + scanpy in the venv)."""
import json
import os
import time
import tempfile

import anndata
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

import featuresel
import hvg
import worker


DENSE = np.array([[5, 1, 0, 7],
                  [0, 0, 3, 7],
                  [2, 0, 0, 0],
                  [0, 4, 4, 0]], dtype=np.float32)


def _write(d, name, X, hvg=True, categorical=False):
    # 4 cells x 4 var rows; rows 1,2 share ENSG2 (duplicate), row 3 is unmapped.
    ids = pd.Series(["ENSG1", "ENSG2", "ENSG2", np.nan], index=[f"g{i}" for i in range(4)])
    var = pd.DataFrame({"gene_id_harmonized": ids.astype("category") if categorical else ids})
    if hvg:
        var["highly_variable"] = [False, False, True, True]
    ad = anndata.AnnData(X=X.copy(), var=var)
    ad.layers["counts"] = X
    p = os.path.join(d, name)
    ad.write_h5ad(p)
    return p


def test_streaming_stats_all_encodings():
    with tempfile.TemporaryDirectory() as d:
        cases = [("csr.h5ad", sparse.csr_matrix(DENSE), False),
                 ("csc.h5ad", sparse.csc_matrix(DENSE), False),
                 ("dense.h5ad", DENSE, False),
                 ("cat.h5ad", sparse.csr_matrix(DENSE), True)]  # categorical var column
        for name, X, cat in cases:
            with h5py.File(_write(d, name, X, categorical=cat), "r") as f:
                g = f["layers/counts"]
                enc = worker._attr(g, "encoding-type", "array")
                keep, codes, ids = worker.harmonized_map(f, 4)
                src = hvg.Reader(g, enc, 4, 4)
                sums, _, det, depth = hvg.moments(src, np.full(4, -1), 1)
                sums, det = sums[0], det[0]
                assert list(depth) == [13, 10, 2, 8], (name, depth)     # per-cell totals
                assert np.allclose(hvg.cell_depth(src), depth), name
                assert (src._sparse_cache or src._dense_cache) is not None   # cached after pass 1
                sums2, _, det2, _ = hvg.moments(src, np.full(4, -1), 1)   # replay from cache
                assert np.allclose(sums2[0], sums) and list(det2[0]) == list(det)
                assert list(ids) == ["ENSG1", "ENSG2"] and int((~keep).sum()) == 1
                # per var row, before merging duplicates
                assert list(det) == [2, 2, 2, 2], (name, det)
                assert list(sums) == [7.0, 5.0, 7.0, 14.0], (name, sums)
                merged = np.bincount(codes, weights=det[keep], minlength=2)
                assert list(merged) == [2, 4], name   # ENSG2 = rows 1+2, capped later
                # upstream flag: either var row HVG -> the merged gene is HVG
                up = np.zeros(2, dtype=bool)
                np.logical_or.at(up, codes, np.asarray(f["var/highly_variable"][:], bool)[keep])
                assert list(up) == [False, True], name


def test_min_datasets_per_category():
    t = pd.DataFrame({
        "biotype": ["protein_coding", "lncRNA", "miRNA", "protein_coding"],
        "n_datasets_hvg": [3, 3, 12, 2],
        "is_mt": [False, False, False, True],
    }, index=list("abcd"))
    for c in featuresel.FLAGCOLS:
        t[c] = t[c] if c in t else False
    rules = {"is_mt": 1, "protein_coding": 3, "lncRNA": 10, "default": 10}
    req = featuresel._min_datasets(t, rules)
    assert list(req) == [3, 10, 10, 1]                      # is_mt wins over protein_coding
    assert list(t["n_datasets_hvg"] >= req) == [True, False, True, True]

    assert list(featuresel._min_datasets(t, {"default": 5})) == [5, 5, 5, 5]
    assert featuresel._parse_min_rules("4", rules) == {"default": 4}
    assert featuresel._parse_min_rules("lncRNA=2", rules)["lncRNA"] == 2


def test_build_rules():
    t = pd.DataFrame({"n_datasets_hvg": [3, 0, 1, 0],
                      "is_pseudogene": [False, False, True, False],
                      "is_mt": [False, True, False, False]},
                     index=["a", "b", "c", "d"])
    keep = featuresel._flag_any(t, ["is_mt"])
    drop = featuresel._flag_any(t, ["is_pseudogene"]) & ~keep
    selected = ((t["n_datasets_hvg"] >= 1) | keep) & ~drop
    assert list(selected) == [True, True, False, False]  # a hvg, b whitelisted, c blacklisted


def test_gene_table():
    df = pd.DataFrame({
        "sample_key": ["s1", "s1", "s2"], "harmonized_id": ["g1", "g2", "g1"],
        "n_cells": [100, 100, 100], "n_detected": [50, 0, 10],
        "sum_counts": [200.0, 0.0, 20.0],
        "score": [2.0, 0.9, 1.6], "rank": [10, 3000, 25],
        "hvg_upstream": [True, False, False]})
    t = featuresel._gene_table(df, s_min=1.5, n_max=3000)
    assert t.loc["g1", "n_datasets_hvg_global"] == 2 and t.loc["g2", "n_datasets_hvg_global"] == 0
    assert t.loc["g1", "best_score"] == 2.0
    assert t.loc["g1", "n_datasets_hvg_upstream"] == 1
    assert t.loc["g1", "pooled_det"] == 60 / 200
    assert t.loc["g1", "mean_counts_per_cell"] == 220 / 200
    assert np.isnan(t.loc["g2", "mean_counts_in_positive"])


def test_union_support_counts_datasets():
    """A dataset votes if the gene is HVG globally OR in >= min_lineages lineages."""
    sub = pd.DataFrame({"sample_key": ["s1", "s1", "s2", "s2"],
                        "harmonized_id": ["g1", "g2", "g1", "g2"],
                        "score": [3.0, 1.0, 1.2, 1.0], "rank": [5, 900, 4000, 800]})
    lin = pd.DataFrame({
        "sample_key": ["s1", "s2", "s2", "s3", "s3", "s2"],
        "lineage":    ["T",  "T",  "B",  "T",  "B",  "T"],
        "lineage_cells": [500, 500, 500, 100, 100, 500],   # s3's lineages are too small
        "harmonized_id": ["g2", "g2", "g2", "g2", "g2", "g1"],
        "score": [2.0, 2.5, 1.8, 9.0, 9.0, 1.4],            # g1 in s2/T fails s_min
        "rank": [10, 8, 30, 1, 1, 100]})
    u = featuresel._union_support(sub, lin, 1.5, 3000, min_cells=200, min_lineages=1)
    assert u["g1"] == 1 and u["g2"] == 2          # g1: s1 global only (s2 rank>n_max, lineage score<s_min)
    u2 = featuresel._union_support(sub, lin, 1.5, 3000, min_cells=200, min_lineages=2)
    assert u2["g2"] == 1                          # only s2 has it in 2 lineages
    assert featuresel._lineage_support(lin, 1.5, 3000, 200, 1)["g2"] == 2
    assert featuresel._lineage_support(lin, 1.5, 3000, 50, 1)["g2"] == 3   # s3 admitted


def test_worker_reads_upstream_hvg():
    with tempfile.TemporaryDirectory() as d:
        h5ad = _write(d, "x.h5ad", sparse.csr_matrix(DENSE))
        out = os.path.join(d, "out.parquet")
        worker.compute({"hvg": {"layer": "counts", "compute": False, "min_cell_counts": 0}},
                       h5ad, "human", "smoke", out)
        got = pd.read_parquet(out).set_index("harmonized_id")
        meta = json.load(open(os.path.splitext(out)[0] + ".meta.json"))
        assert list(got.index) == ["ENSG1", "ENSG2"]
        assert list(got["hvg_upstream"]) == [False, True]
        assert list(got["n_detected"]) == [2, 4] and list(got["sum_counts"]) == [7.0, 12.0]
        assert (got["rank"] == 0).all() and got["score"].isna().all()   # nothing computed
        assert meta["n_duplicate_harmonized_ids"] == 1
        assert meta["upstream_hvg"]["n"] == 1

        # no var.highly_variable in the file -> flag stays False, recorded as None
        worker.compute({"hvg": {"layer": "counts", "compute": False, "min_cell_counts": 0}},
                       _write(d, "nohvg.h5ad", sparse.csr_matrix(DENSE), hvg=False),
                       "human", "smoke2", out)
        assert json.load(open(os.path.splitext(out)[0] + ".meta.json"))["upstream_hvg"] is None


def test_computed_hvg_end_to_end():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 200, 300
    counts = rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)
    counts[:, :20] += rng.poisson(20, size=(n_cells, 20))  # a few loud genes
    ids = [f"ENSG{i}" for i in range(n_genes - 1)] + ["ENSG0"]  # last col duplicates the first
    ad = anndata.AnnData(X=sparse.csr_matrix(counts),
                         var=pd.DataFrame({"gene_id_harmonized": ids}))
    ad.layers["counts"] = ad.X.copy()
    with tempfile.TemporaryDirectory() as d:
        h5ad = os.path.join(d, "x.h5ad")
        ad.write_h5ad(h5ad)
        cfg = {"hvg": {"compute": True, "layer": "counts", "min_cell_counts": 0}}
        out = os.path.join(d, "out.parquet")
        worker.compute(cfg, h5ad, "human", "smoke", out)
        got = pd.read_parquet(out)
        meta = json.load(open(os.path.splitext(out)[0] + ".meta.json"))
        assert len(got) == n_genes - 1              # duplicate merged
        assert (got["rank"] > 0).sum() == meta["n_rankable"][0]
        assert (got["rank"] <= 50).sum() == 50      # ranks are 1..n, dense
        assert got.sort_values("rank").iloc[0]["score"] == got["score"].max()
        assert got["n_detected"].max() <= n_cells
        assert not got["symbol"].isna().any()


def _hvg_case(X, name, d):
    """Run our streaming seurat_v3 on the whole matrix, via a written h5ad."""
    ad = anndata.AnnData(X=sparse.csr_matrix(X))
    ad.layers["counts"] = ad.X
    p = os.path.join(d, name)
    ad.write_h5ad(p)
    return p


def test_streaming_seurat_v3_matches_scanpy():
    """Our two-pass implementation must reproduce scanpy's seurat_v3 ranking."""
    import scanpy as sc
    rng = np.random.default_rng(3)
    n_cells, n_genes, n_top = 400, 500, 100
    X = rng.poisson(0.4, size=(n_cells, n_genes)).astype(np.float32)
    X[:, :40] += rng.poisson(15, size=(n_cells, 40))
    X[:150, 40:80] += rng.poisson(25, size=(150, 40))   # a subpopulation
    with tempfile.TemporaryDirectory() as d:
        with h5py.File(_hvg_case(X, "x.h5ad", d), "r") as f:
            g = f["layers/counts"]
            out, ndet, sums, _, _, valid = hvg.call(hvg.Reader(g, "csr_matrix", n_cells, n_genes),
                                                    np.full(n_cells, -1), np.array([n_cells]),
                                                    methods=("vst",), min_gene_cells=0)
            flags = hvg.top_n(out["vst"], valid, n_top)
            ndet, sums = ndet[0], sums[0]
        ours = set(np.flatnonzero(flags[0]))
        ad = anndata.AnnData(X=sparse.csr_matrix(X))
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top, flavor="seurat_v3")
        theirs = set(np.flatnonzero(np.asarray(ad.var["highly_variable"], bool)))
        jac = len(ours & theirs) / len(ours | theirs)
        assert jac > 0.95, f"jaccard vs scanpy = {jac:.3f}"
        assert list(ndet) == list((X > 0).sum(axis=0)), "detection counts"
        assert np.allclose(sums, X.sum(axis=0), rtol=1e-5), "count sums"


def test_lineage_hvg_finds_subpopulation_genes():
    """Genes variable only inside a minority lineage must be missed by the
    dataset-level call and caught by that lineage's own call."""
    rng = np.random.default_rng(4)
    n_cells, n_genes, n_top = 600, 400, 60
    X = rng.poisson(0.4, size=(n_cells, n_genes)).astype(np.float32)
    X[:, :50] += rng.poisson(20, size=(n_cells, 50))          # loud everywhere
    minority = np.zeros(n_cells, bool); minority[:120] = True  # 20% of cells
    X[np.ix_(minority, np.arange(300, 340))] += rng.poisson(1, size=(120, 40)) * \
        rng.integers(0, 30, size=(120, 40))                    # variable only here
    grp = np.where(minority, 0, 1).astype(np.int64)
    sizes = np.array([n_cells, int(minority.sum()), int((~minority).sum())])
    with tempfile.TemporaryDirectory() as d:
        with h5py.File(_hvg_case(X, "x.h5ad", d), "r") as f:
            out, _, _, _, _, valid = hvg.call(hvg.Reader(f["layers/counts"], "csr_matrix", n_cells, n_genes),
                                              grp, sizes, methods=("vst",))
            flags = hvg.top_n(out["vst"], valid, n_top)
    target = np.arange(300, 340)
    caught_global = flags[0, target].sum()
    caught_minor = flags[1, target].sum()
    assert caught_minor > caught_global, (caught_minor, caught_global)
    union = flags.any(axis=0)
    assert union.sum() > flags[0].sum()          # the union is strictly larger
    assert union[flags[0]].all()                 # and contains the global call


def test_pearson_matches_scanpy():
    """Streamed analytic Pearson residual variance == scanpy experimental."""
    import scanpy as sc
    rng = np.random.default_rng(5)
    n_cells, n_genes, n_top = 300, 400, 80
    depth = rng.integers(500, 5000, size=n_cells)                 # uneven sequencing depth
    p = rng.dirichlet(np.full(n_genes, 0.3))
    X = rng.poisson(depth[:, None] * p[None, :]).astype(np.float32)
    X[:100, :30] += rng.poisson(8, size=(100, 30))                 # a subpopulation
    with tempfile.TemporaryDirectory() as d:
        for enc_name, Xw in [("csr", sparse.csr_matrix(X)), ("dense", X)]:
            with h5py.File(_hvg_case(Xw, f"{enc_name}.h5ad", d), "r") as f:
                g = f["layers/counts"]
                enc = worker._attr(g, "encoding-type", "array")
                out, *_, valid = hvg.call(hvg.Reader(g, enc, n_cells, n_genes), np.full(n_cells, -1),
                                          np.array([n_cells]), methods=("pearson",), min_gene_cells=0)
            ad = anndata.AnnData(X=sparse.csr_matrix(X))
            sc.experimental.pp.highly_variable_genes(ad, flavor="pearson_residuals", n_top_genes=n_top)
            ref = ad.var["residual_variances"].to_numpy()
            assert np.allclose(out["pearson"][0][valid[0]], ref[valid[0]], rtol=1e-4), enc_name
            ours = set(np.flatnonzero(hvg.top_n(out["pearson"], valid, n_top)[0]))
            theirs = set(np.flatnonzero(ad.var["highly_variable"].to_numpy()))
            assert len(ours & theirs) / len(ours | theirs) > 0.95, enc_name


def test_dropped_cells_and_gene_guard():
    """grp == -2 cells vanish from every group; sparse genes are not rankable."""
    rng = np.random.default_rng(6)
    n_cells, n_genes = 120, 200
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)
    X[:20] = 0                      # empty cells
    X[:, 0] = 0; X[5, 0] = 7        # gene 0 seen in one (dropped) cell only
    X[:, 1] = 0; X[50, 1] = 3       # gene 1 seen in exactly one kept cell
    grp = np.full(n_cells, -1); grp[:20] = -2
    with tempfile.TemporaryDirectory() as d:
        with h5py.File(_hvg_case(X, "x.h5ad", d), "r") as f:
            g = f["layers/counts"]
            out, ndet, sums, mean, var, valid = hvg.call(
                hvg.Reader(g, "csr_matrix", n_cells, n_genes), grp, np.array([100]),
                methods=("vst", "pearson"), min_gene_cells=3)
    assert ndet[0, 0] == 0 and sums[0, 0] == 0           # dropped cell's count is gone
    assert ndet[0, 1] == 1 and not valid[0, 1]           # 1 < min_gene_cells -> unrankable
    assert out["vst"][0, 1] == 0 and out["pearson"][0, 1] == 0
    assert hvg.ranks(out["vst"], valid)[0, 1] == 0
    assert valid[0, 2:].all()


def test_expected_sd_falls_back_when_loess_fails():
    mean = np.concatenate([np.zeros(50), np.linspace(0.01, 5, 100)])
    var = np.concatenate([np.zeros(50), np.linspace(0.01, 9, 100)])
    notes = []
    sd, ok = hvg.expected_sd(mean, var, np.ones(150, bool), 0.3, notes)
    assert (sd[ok] > 0).all() and not ok[:50].any()
    notes = []
    sd, ok = hvg.expected_sd(np.zeros(200), np.zeros(200), np.ones(200, bool), 0.3, notes)
    assert not ok.any() and notes  # degenerate input reports, does not raise




def _slow_or_fast(x):
    import time
    if x == "slow":
        time.sleep(30)
    return x.upper()


def test_fs_parallel_defers_hung_calls():
    t0 = time.time()
    res, deferred = featuresel.fs_parallel(_slow_or_fast, ["a", "slow", "b"], timeout=2)
    assert res == {"a": "A", "b": "B"}, res
    assert deferred == ["slow"], deferred
    assert time.time() - t0 < 10          # the deadline held; the sleeper was abandoned
    print("[fs_parallel] fast items returned, slow one deferred")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    featuresel.fs_exit(0)   # do not wait for the deliberately hung child
