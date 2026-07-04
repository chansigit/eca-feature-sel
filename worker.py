#!/usr/bin/env python
"""eca-feature-sel Stage-1 worker.

Computes per-gene detection stats from ONE h5ad's raw ``layers/counts`` via
streaming h5py (CSR or CSC), WITHOUT densifying or building an AnnData. Emits,
per harmonized gene (``var.gene_id_harmonized`` = Ensembl ENSG/ENSMUSG):

    n_cells, n_detected (cells with count>0), sum_counts

Unmapped var rows (no shared cross-dataset key) are counted only in the meta.

Run per Slurm array task:
    worker.py --manifest jobs/manifest.tsv --index $SLURM_ARRAY_TASK_ID
or directly:
    worker.py --h5ad X.h5ad --species human --sample-key ts-lung --out out.parquet
"""
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd

CHUNK = 20_000_000  # nnz read block; caps memory on the largest (9.6 GB) files
UNION_CHUNK = 5_000_000  # smaller block for duplicate-gene union detection
MISSING_IDS = {"", "nan", "na", "none", "null"}


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr], dtype=object)


def _attr(node, name, default=b""):
    v = node.attrs.get(name, default)
    return v.decode() if isinstance(v, bytes) else v


def _normalize_ids(raw):
    hid = np.empty(len(raw), dtype=object)
    hid[:] = None
    mapped = np.zeros(len(raw), dtype=bool)
    for i, x in enumerate(raw):
        if x is None:
            continue
        s = str(x).strip()
        if s.lower() in MISSING_IDS:
            continue
        hid[i] = s
        mapped[i] = True
    return hid, mapped


def _csr_gene_stats(g, n_vars):
    """CSR (cells x genes): detection & sum per gene via chunked bincount over nnz."""
    indices, data = g["indices"], g["data"]
    nnz = int(indices.shape[0])
    n_det = np.zeros(n_vars, dtype=np.int64)
    sum_c = np.zeros(n_vars, dtype=np.float64)
    for s in range(0, nnz, CHUNK):
        e = min(s + CHUNK, nnz)
        idx = np.asarray(indices[s:e])
        dat = np.asarray(data[s:e], dtype=np.float64)
        n_det += np.bincount(idx, minlength=n_vars)[:n_vars].astype(np.int64)
        sum_c += np.bincount(idx, weights=dat, minlength=n_vars)[:n_vars]
    return n_det, sum_c


def _csc_gene_stats(g, n_vars):
    """CSC (cells x genes): genes are columns; detection = nnz per column =
    diff(indptr) (no data read); sum via a cumulative-sum difference."""
    indptr = np.asarray(g["indptr"][:])
    if len(indptr) != n_vars + 1:
        raise SystemExit(f"csc indptr len {len(indptr)} != n_vars+1 {n_vars + 1}")
    n_det = np.diff(indptr).astype(np.int64)
    data = np.asarray(g["data"][:], dtype=np.float64)
    cs = np.empty(len(data) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(data, out=cs[1:])
    sum_c = cs[indptr[1:]] - cs[indptr[:-1]]
    return n_det, sum_c


def harmonized_ids(f, n_vars):
    node = f["var/gene_id_harmonized"]
    if isinstance(node, h5py.Group):  # AnnData categorical: categories + codes
        cats = _decode(node["categories"][:])
        codes = np.asarray(node["codes"][:])
        raw = np.empty(n_vars, dtype=object)
        raw[:] = None
        coded = codes >= 0
        raw[coded] = cats[codes[coded]]
    else:  # plain string dataset fallback
        raw = _decode(node[:])
    return _normalize_ids(raw)


def _group_ids(hid, mapped):
    ids = np.array(sorted(set(hid[mapped])), dtype=object)
    var_to_group = np.full(len(hid), -1, dtype=np.int64)
    if len(ids):
        lookup = {gid: i for i, gid in enumerate(ids)}
        for i in np.flatnonzero(mapped):
            var_to_group[i] = lookup[hid[i]]
    group_sizes = np.bincount(var_to_group[mapped], minlength=len(ids)) if len(ids) else np.array([], dtype=np.int64)
    return ids, var_to_group, group_sizes


def _row_blocks(indptr, max_nnz):
    n_obs = len(indptr) - 1
    start = 0
    while start < n_obs:
        limit = indptr[start] + max_nnz
        end = int(np.searchsorted(indptr, limit, side="right") - 1)
        end = max(start + 1, min(end, n_obs))
        yield start, end
        start = end


def _csr_duplicate_detection(g, var_to_dup, n_dup):
    """Exact detection for duplicated harmonized IDs: count unique cell/gene pairs."""
    indptr = np.asarray(g["indptr"][:])
    indices = g["indices"]
    dup_det = np.zeros(n_dup, dtype=np.int64)
    for r0, r1 in _row_blocks(indptr, UNION_CHUNK):
        s, e = int(indptr[r0]), int(indptr[r1])
        if e <= s:
            continue
        idx = np.asarray(indices[s:e])
        dup = var_to_dup[idx]
        keep = dup >= 0
        if not keep.any():
            continue
        rows = np.repeat(np.arange(r1 - r0, dtype=np.int64), np.diff(indptr[r0:r1 + 1]))[keep]
        keys = rows * n_dup + dup[keep]
        uniq = np.unique(keys)
        dup_det += np.bincount(uniq % n_dup, minlength=n_dup).astype(np.int64)
    return dup_det


def _csc_duplicate_detection(g, var_to_dup, n_dup):
    indptr = np.asarray(g["indptr"][:])
    indices = g["indices"]
    cells_by_dup = [[] for _ in range(n_dup)]
    for j in np.flatnonzero(var_to_dup >= 0):
        s, e = int(indptr[j]), int(indptr[j + 1])
        if e > s:
            cells_by_dup[int(var_to_dup[j])].append(np.asarray(indices[s:e], dtype=np.int64))
    dup_det = np.zeros(n_dup, dtype=np.int64)
    for i, arrays in enumerate(cells_by_dup):
        if not arrays:
            continue
        cells = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
        dup_det[i] = int(np.unique(cells).size)
    return dup_det


def compute(h5ad, species, key, out):
    with h5py.File(h5ad, "r") as f:
        g = f["layers/counts"]
        enc = _attr(g, "encoding-type")
        n_obs, n_vars = (int(x) for x in g.attrs["shape"])  # cells, genes
        if enc == "csr_matrix":
            n_det, sum_c = _csr_gene_stats(g, n_vars)
        elif enc == "csc_matrix":
            n_det, sum_c = _csc_gene_stats(g, n_vars)
        else:
            raise SystemExit(f"{key}: layers/counts encoding={enc!r} (need csr/csc)")
        hid, mapped = harmonized_ids(f, n_vars)
        ids, var_to_group, group_sizes = _group_ids(hid, mapped)
        group_idx = var_to_group[mapped]
        sum_counts = np.bincount(group_idx, weights=sum_c[mapped], minlength=len(ids))
        n_detected = np.bincount(group_idx, weights=n_det[mapped], minlength=len(ids)).astype(np.int64)

        dup_groups = np.flatnonzero(group_sizes > 1)
        if len(dup_groups):
            dup_lookup = {gid: i for i, gid in enumerate(dup_groups)}
            var_to_dup = np.full(n_vars, -1, dtype=np.int64)
            for i in np.flatnonzero(mapped):
                gid = int(var_to_group[i])
                if gid in dup_lookup:
                    var_to_dup[i] = dup_lookup[gid]
            if enc == "csr_matrix":
                dup_det = _csr_duplicate_detection(g, var_to_dup, len(dup_groups))
            else:
                dup_det = _csc_duplicate_detection(g, var_to_dup, len(dup_groups))
            n_detected[dup_groups] = dup_det

    if len(hid) != n_vars:
        raise SystemExit(f"{key}: var length {len(hid)} != counts n_vars {n_vars}")

    agg = pd.DataFrame({"harmonized_id": ids, "n_detected": n_detected, "sum_counts": sum_counts})
    agg.insert(0, "sample_key", key)
    agg.insert(1, "species", species)
    agg["n_cells"] = int(n_obs)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    agg.to_parquet(tmp, index=False)
    os.replace(tmp, out)  # atomic: a half-written parquet never looks "done"

    meta = {
        "sample_key": key, "species": species, "h5ad": h5ad,
        "h5ad_mtime": os.path.getmtime(h5ad), "encoding": enc,
        "n_cells": int(n_obs), "n_vars_original": int(n_vars),
        "n_genes_harmonized": int(agg.shape[0]),
        "n_unmapped_rows": int((~mapped).sum()),
        "n_unmapped_detected_rows": int((n_det[~mapped] > 0).sum()),
        "n_duplicate_harmonized_ids": int(len(dup_groups)),
        "nnz": int(n_det.sum()),
    }
    with open(os.path.splitext(out)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{key}] {species} cells={n_obs} genes={agg.shape[0]} enc={enc} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="eca-feature-sel stage-1 worker")
    ap.add_argument("--manifest")
    ap.add_argument("--index", type=int, help="1-based line in manifest")
    ap.add_argument("--h5ad")
    ap.add_argument("--species")
    ap.add_argument("--sample-key")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.manifest:
        lines = [ln for ln in open(a.manifest).read().splitlines() if ln.strip()]
        key, species, h5ad, out = lines[a.index - 1].split("\t")
    else:
        key, species, h5ad, out = a.sample_key, a.species, a.h5ad, a.out
    compute(h5ad, species, key, out)


if __name__ == "__main__":
    main()
