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


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr], dtype=object)


def _attr(node, name, default=b""):
    v = node.attrs.get(name, default)
    return v.decode() if isinstance(v, bytes) else v


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
    hid = np.empty(n_vars, dtype=object)
    hid[:] = None
    if isinstance(node, h5py.Group):  # AnnData categorical: categories + codes
        cats = _decode(node["categories"][:])
        codes = np.asarray(node["codes"][:])
        mapped = codes >= 0
        hid[mapped] = cats[codes[mapped]]
    else:  # plain string dataset fallback
        raw = _decode(node[:])
        mapped = np.array([x not in (None, "", "nan", "NA") for x in raw])
        hid[mapped] = raw[mapped]
    return hid, mapped


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

    if len(hid) != n_vars:
        raise SystemExit(f"{key}: var length {len(hid)} != counts n_vars {n_vars}")

    df = pd.DataFrame({"harmonized_id": hid, "n_detected": n_det, "sum_counts": sum_c})
    agg = (
        df[df["harmonized_id"].notna()]
        .groupby("harmonized_id", as_index=False)
        .agg(n_detected=("n_detected", "sum"), sum_counts=("sum_counts", "sum"))
    )
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
