#!/usr/bin/env python
"""eca-feature-sel Stage-2 worker: per-dataset HVG and cluster marker genes."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import featuresel  # noqa: E402

MISSING_IDS = {"", "nan", "na", "none", "null"}
MISSING_LABELS = {"", "nan", "na", "none", "null", "unknown", "unassigned"}


def _normalize_ids(values):
    out = []
    keep = []
    for x in values:
        if pd.isna(x):
            out.append(None)
            keep.append(False)
            continue
        s = str(x).strip()
        valid = s.lower() not in MISSING_IDS
        out.append(s if valid else None)
        keep.append(valid)
    return np.array(out, dtype=object), np.array(keep, dtype=bool)


def _choose_obs_key(adata, keys):
    for key in keys:
        if key in adata.obs:
            return key
    return None


def _valid_labels(series):
    vals = series.astype("object")
    labels = vals.map(lambda x: "" if pd.isna(x) else str(x).strip())
    return labels, ~labels.str.lower().isin(MISSING_LABELS)


def _frac_nonzero(x):
    nz = x > 0
    if sparse.issparse(nz):
        return np.asarray(nz.mean(axis=0)).ravel()
    return np.asarray(nz).mean(axis=0)


def _rank_genes_array(uns, key, group):
    arr = uns.get(key)
    if arr is None:
        return None
    if getattr(arr.dtype, "names", None):
        return np.asarray(arr[group])
    return np.asarray(pd.DataFrame(arr)[group])


def _run_hvg(adata, cfg, meta):
    rule = cfg["select_v2"]["hvg_rule"]
    layer = rule.get("layer", "counts")
    if layer not in adata.layers:
        meta["hvg_error"] = f"layer {layer!r} not found"
        return pd.DataFrame(columns=["harmonized_id", "hvg_best_rank"])

    n_top = min(int(rule.get("n_top_genes", 3000)), adata.n_vars)
    if n_top <= 0 or adata.n_obs == 0 or adata.n_vars == 0:
        return pd.DataFrame(columns=["harmonized_id", "hvg_best_rank"])

    try:
        sc.pp.highly_variable_genes(
            adata,
            layer=layer,
            n_top_genes=n_top,
            flavor=rule.get("flavor", "seurat_v3"),
            subset=False,
            inplace=True,
        )
    except Exception as e:
        meta["hvg_error"] = repr(e)
        return pd.DataFrame(columns=["harmonized_id", "hvg_best_rank"])

    hv = adata.var.get("highly_variable")
    if hv is None:
        return pd.DataFrame(columns=["harmonized_id", "hvg_best_rank"])
    hv = hv.fillna(False).astype(bool).to_numpy()
    if not hv.any():
        return pd.DataFrame(columns=["harmonized_id", "hvg_best_rank"])

    ranks = adata.var.get("highly_variable_rank")
    if ranks is None:
        ranks = pd.Series(np.arange(adata.n_vars), index=adata.var_names)
    hvg = pd.DataFrame({
        "harmonized_id": adata.var["_efs_harmonized_id"].to_numpy()[hv],
        "hvg_best_rank": pd.to_numeric(ranks, errors="coerce").to_numpy()[hv],
    })
    out = hvg.groupby("harmonized_id", observed=True).agg(hvg_best_rank=("hvg_best_rank", "min"))
    out["selected_hvg"] = True
    return out.reset_index()


def _run_cluster_deg(adata, cfg, meta):
    rule = cfg["select_v2"]["cluster_deg_rule"]
    obs_key = _choose_obs_key(adata, rule.get("obs_key_priority", []))
    meta["deg_obs_key"] = obs_key
    if obs_key is None:
        meta["deg_status"] = "missing_obs_key"
        return pd.DataFrame(columns=["harmonized_id"])

    labels, valid = _valid_labels(adata.obs[obs_key])
    if not valid.any():
        meta["deg_status"] = "no_valid_labels"
        return pd.DataFrame(columns=["harmonized_id"])

    ad = adata[valid.to_numpy()].copy()
    labels = labels[valid].reset_index(drop=True)
    ad.obs["_efs_group"] = pd.Categorical(labels.to_numpy())
    counts = pd.Series(ad.obs["_efs_group"]).value_counts()
    min_cluster = int(rule.get("min_cluster_cells", 50))
    min_other = int(rule.get("min_other_cells", 50))
    groups = [g for g, n in counts.items() if n >= min_cluster and (ad.n_obs - n) >= min_other]
    meta["deg_n_groups_total"] = int(counts.shape[0])
    meta["deg_n_groups_tested"] = int(len(groups))
    if not groups:
        meta["deg_status"] = "no_eligible_groups"
        return pd.DataFrame(columns=["harmonized_id"])

    layer = cfg["select_v2"]["hvg_rule"].get("layer", "counts")
    if layer not in ad.layers:
        meta["deg_error"] = f"layer {layer!r} not found"
        return pd.DataFrame(columns=["harmonized_id"])
    ad.X = ad.layers[layer].copy()
    ad.uns.pop("log1p", None)
    sc.pp.normalize_total(ad, target_sum=cfg["select_v2"]["hvg_rule"].get("normalize_target_sum", 10000))
    sc.pp.log1p(ad)

    top_n = int(rule.get("top_n_per_cluster", 100))
    multiplier = int(rule.get("candidate_multiplier", 20))
    n_genes = min(ad.n_vars, max(top_n, top_n * multiplier))
    try:
        sc.tl.rank_genes_groups(
            ad,
            groupby="_efs_group",
            groups=[str(g) for g in groups],
            reference="rest",
            method=rule.get("method", "wilcoxon"),
            n_genes=n_genes,
            use_raw=False,
        )
    except Exception as e:
        meta["deg_error"] = repr(e)
        return pd.DataFrame(columns=["harmonized_id"])

    uns = ad.uns["rank_genes_groups"]
    name_to_idx = pd.Series(np.arange(ad.n_vars), index=ad.var_names)
    rows = []
    for group in groups:
        group = str(group)
        names = _rank_genes_array(uns, "names", group)
        if names is None or len(names) == 0:
            continue
        logfc = _rank_genes_array(uns, "logfoldchanges", group)
        pvals_adj = _rank_genes_array(uns, "pvals_adj", group)
        scores = _rank_genes_array(uns, "scores", group)
        idx = name_to_idx.reindex(names).dropna().astype(int).to_numpy()
        if idx.size == 0:
            continue

        in_mask = (ad.obs["_efs_group"].astype(str).to_numpy() == group)
        pct_in = _frac_nonzero(ad.X[in_mask][:, idx])
        pct_out = _frac_nonzero(ad.X[~in_mask][:, idx])
        pct_diff = pct_in - pct_out
        logfc = np.asarray(logfc[:idx.size], dtype=float) if logfc is not None else np.full(idx.size, np.nan)
        pvals_adj = np.asarray(pvals_adj[:idx.size], dtype=float) if pvals_adj is not None else np.full(idx.size, np.nan)
        scores = np.asarray(scores[:idx.size], dtype=float) if scores is not None else np.full(idx.size, np.nan)

        keep = (
            (pct_in >= float(rule.get("min_pct_in_cluster", 0.10))) &
            (pct_diff >= float(rule.get("min_pct_difference", 0.05))) &
            (logfc >= float(rule.get("min_log2fc", 0.25))) &
            (pvals_adj <= float(rule.get("max_adj_pvalue", 0.05)))
        )
        kept = 0
        for rank, (var_idx, ok) in enumerate(zip(idx, keep), 1):
            if not ok:
                continue
            rows.append({
                "harmonized_id": ad.var["_efs_harmonized_id"].iat[var_idx],
                "deg_group": group,
                "deg_rank": rank,
                "deg_score": scores[rank - 1],
                "deg_log2fc": logfc[rank - 1],
                "deg_adj_pvalue": pvals_adj[rank - 1],
                "deg_pct_in_cluster": pct_in[rank - 1],
                "deg_pct_out_cluster": pct_out[rank - 1],
            })
            kept += 1
            if kept >= top_n:
                break

    if not rows:
        meta["deg_status"] = "no_markers_after_filters"
        return pd.DataFrame(columns=["harmonized_id"])

    deg = pd.DataFrame(rows)
    out = deg.groupby("harmonized_id", observed=True).agg(
        n_deg_groups=("deg_group", "nunique"),
        best_deg_rank=("deg_rank", "min"),
        best_deg_score=("deg_score", "max"),
        best_deg_log2fc=("deg_log2fc", "max"),
        best_deg_adj_pvalue=("deg_adj_pvalue", "min"),
        max_deg_pct_in_cluster=("deg_pct_in_cluster", "max"),
        max_deg_pct_out_cluster=("deg_pct_out_cluster", "max"),
    )
    out["selected_cluster_deg"] = True
    meta["deg_status"] = "ok"
    meta["deg_n_marker_rows"] = int(len(deg))
    meta["deg_n_marker_genes"] = int(len(out))
    return out.reset_index()


def compute(cfg, h5ad, species, key, out):
    meta = {
        "sample_key": key,
        "species": species,
        "h5ad": h5ad,
        "h5ad_mtime": os.path.getmtime(h5ad),
        "selection_v2_measure_signature": featuresel.selection_v2_measure_signature(cfg),
    }
    gene_key = "gene_id_harmonized"
    adata = sc.read_h5ad(h5ad)
    if gene_key not in adata.var:
        raise SystemExit(f"{key}: var.{gene_key} not found")
    hids, mapped = _normalize_ids(adata.var[gene_key].to_numpy())
    meta["n_cells"] = int(adata.n_obs)
    meta["n_vars_original"] = int(adata.n_vars)
    meta["n_unmapped_rows"] = int((~mapped).sum())
    if not mapped.any():
        raise SystemExit(f"{key}: no valid harmonized gene ids")

    adata = adata[:, mapped].copy()
    adata.var["_efs_harmonized_id"] = hids[mapped]
    adata.var_names = [f"var_{i}" for i in range(adata.n_vars)]

    hvg = _run_hvg(adata, cfg, meta)
    deg = _run_cluster_deg(adata, cfg, meta)
    hvg = hvg.set_index("harmonized_id") if "harmonized_id" in hvg else pd.DataFrame()
    deg = deg.set_index("harmonized_id") if "harmonized_id" in deg else pd.DataFrame()
    idx = hvg.index.union(deg.index)
    res = pd.DataFrame(index=idx)
    if not hvg.empty:
        res = res.join(hvg, how="left")
    if not deg.empty:
        res = res.join(deg, how="left")
    for c in ["selected_hvg", "selected_cluster_deg"]:
        res[c] = res[c].fillna(False).astype(bool) if c in res else False
    res = res[(res["selected_hvg"]) | (res["selected_cluster_deg"])].reset_index()
    if "index" in res.columns:
        res = res.rename(columns={"index": "harmonized_id"})
    res.insert(0, "sample_key", key)
    res.insert(1, "species", species)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    res.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    meta["n_genes_harmonized"] = int(len(set(adata.var["_efs_harmonized_id"])))
    meta["n_hvg_genes"] = int(res["selected_hvg"].sum()) if "selected_hvg" in res else 0
    meta["n_cluster_deg_genes"] = int(res["selected_cluster_deg"].sum()) if "selected_cluster_deg" in res else 0
    with open(os.path.splitext(out)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{key}] {species} hvg={meta['n_hvg_genes']} cluster_deg={meta['n_cluster_deg_genes']} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="eca-feature-sel stage-2 worker")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--manifest")
    ap.add_argument("--index", type=int, help="1-based line in manifest")
    ap.add_argument("--h5ad")
    ap.add_argument("--species")
    ap.add_argument("--sample-key")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = featuresel.load_config(a.config)
    if a.manifest:
        lines = [ln for ln in open(a.manifest).read().splitlines() if ln.strip()]
        key, species, h5ad, out = lines[a.index - 1].split("\t")
    else:
        key, species, h5ad, out = a.sample_key, a.species, a.h5ad, a.out
    compute(cfg, h5ad, species, key, out)


if __name__ == "__main__":
    main()
