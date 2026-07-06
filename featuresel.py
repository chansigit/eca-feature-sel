#!/usr/bin/env python
"""eca-feature-sel: counts-driven gene-vocabulary selection for single-cell FMs.

Measurement (Stage 1/2, Slurm-parallel, cached) is separated from policy
(Stage 3, instant, re-runnable). Human and mouse are built independently in
their own Ensembl ID spaces. Category rules are applied only after data-driven
candidate selection.

Subcommands:
  status               scan corpus vs cache; report done / stale / missing
  measure [--force]    submit a Slurm job array for stale/missing datasets (reuses the rest)
  measure-v2 [--force] submit HVG/cluster-DEG measurement jobs
  ref [--force]        build the Ensembl biotype reference (needs internet)
  build [opts] [--tag] aggregate cached stats + select -> a versioned vocab snapshot
  build-v2 [opts]      build detection + HVG + cluster-DEG vocabulary
  refresh [opts]       measure -> wait -> build (one shot)
  refresh-v2 [opts]    measure + measure-v2 -> wait -> build-v2
  diff A B             compare two vocab snapshots (genes added / removed)
  list                 list vocab snapshots

Staleness = stage-1 output missing OR h5ad newer than it. So re-running only
recomputes changed/new datasets; everything else is reused.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request

import h5py
import numpy as np
import pandas as pd
import yaml

pd.set_option("future.no_silent_downcasting", True)  # quiet fillna(False) on flag cols

HERE = os.path.dirname(os.path.abspath(__file__))
DET_THRESH = [0.005, 0.01, 0.02, 0.05]
NARROW = ["is_OR", "is_vomeronasal", "is_taste",
          "is_IG_V", "is_IG_D", "is_IG_J", "is_TR_V", "is_TR_D", "is_TR_J"]
FLAGCOLS = NARROW + ["is_pseudogene", "is_IG_C", "is_TR_C", "is_mt", "is_hb", "is_ribo", "is_sex"]
SPECIES = ("human", "mouse")


# ---------------------------------------------------------------- config / paths
def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = path
    c = cfg["cache_root"]
    cfg["_dirs"] = {d: os.path.join(c, d) for d in
                    ("stage1", "stage2", "ref", "master", "vocab", "jobs")}
    cfg["_dirs"]["logs"] = os.path.join(c, "jobs", "logs")
    for d in cfg["_dirs"].values():
        os.makedirs(d, exist_ok=True)
    return cfg


def _read_stats(dsdir):
    js = os.path.join(dsdir, "curation_stats.json")
    if not os.path.exists(js):
        return {}
    try:
        with open(js) as fh:
            return json.load(fh)
    except Exception:
        return {}


def species_of(dsdir, stats=None):
    """Prefer curation_stats.json species; fall back to the dir-name prefix."""
    sp = (stats or _read_stats(dsdir)).get("species")
    if sp in SPECIES:
        return sp
    b = os.path.basename(dsdir)
    if b.startswith(("ts-", "3ca-")):
        return "human"
    if b.startswith(("tm-", "tms-")):
        return "mouse"
    return None


def _decode_attr_list(v):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in v]


def _h5ad_info(path):
    """Return minimal content metadata for h5ad selection, without loading AnnData."""
    try:
        with h5py.File(path, "r") as f:
            if "layers/counts" not in f or "var/gene_id_harmonized" not in f:
                return None
            shape = f["layers/counts"].attrs.get("shape")
            if shape is None or len(shape) != 2:
                return None
            obs_cols = set(_decode_attr_list(f["obs"].attrs.get("column-order", []))) if "obs" in f else set()
            post_filter_cols = {"doublet_score", "predicted_doublet", "mrvi_leiden", "harmony_leiden"}
            return {
                "path": path,
                "n_obs": int(shape[0]),
                "n_vars": int(shape[1]),
                "has_post_filter_cols": bool(obs_cols & post_filter_cols),
            }
    except Exception:
        return None


def choose_h5ad(dsdir, stats):
    """Choose the dataset h5ad by content, not by filename convention.

    Prefer files that are usable by worker.py and whose shape matches
    curation_stats.json. If multiple files remain equivalent, use content-derived
    post-filter columns as a tie-breaker, then fall back to mtime/path for a
    deterministic choice.
    """
    candidates = []
    for name in sorted(os.listdir(dsdir)):
        path = os.path.join(dsdir, name)
        if not os.path.isfile(path) or not name.lower().endswith(".h5ad"):
            continue
        info = _h5ad_info(path)
        if info is None:
            continue
        score = 0
        if stats.get("cells_kept") is not None and info["n_obs"] == int(stats["cells_kept"]):
            score += 4
        if stats.get("genes") is not None and info["n_vars"] == int(stats["genes"]):
            score += 2
        if info["has_post_filter_cols"]:
            score += 1
        candidates.append((score, os.path.getmtime(path), path))
    if not candidates:
        return None
    return max(candidates)[2]


def _resolve_path(path, base):
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))


def _stage1_record(cfg, key, sp, h5ad):
    if sp not in SPECIES:
        raise SystemExit(f"{key}: species={sp!r} (expected one of {SPECIES})")
    if not os.path.exists(h5ad):
        raise SystemExit(f"{key}: h5ad not found: {h5ad}")
    par = os.path.join(cfg["_dirs"]["stage1"], key + ".parquet")
    stale = (not os.path.exists(par)) or (os.path.getmtime(h5ad) > os.path.getmtime(par))
    return {"key": key, "species": sp, "h5ad": h5ad, "out": par, "stale": stale}


def selection_v2_measure_signature(cfg):
    d = cfg.get("select_v2", {})
    payload = {
        "hvg_rule": d.get("hvg_rule", {}),
        "cluster_deg_rule": d.get("cluster_deg_rule", {}),
    }
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode()).hexdigest()


def _stage2_record(cfg, rec):
    out = os.path.join(cfg["_dirs"]["stage2"], rec["key"] + ".parquet")
    meta = os.path.splitext(out)[0] + ".meta.json"
    sig = selection_v2_measure_signature(cfg)
    stale = (not os.path.exists(out)) or (os.path.getmtime(rec["h5ad"]) > os.path.getmtime(out))
    if os.path.exists(meta):
        try:
            stale = stale or json.load(open(meta)).get("selection_v2_measure_signature") != sig
        except Exception:
            stale = True
    else:
        stale = True
    r = dict(rec)
    r["stage2_out"] = out
    r["stage2_stale"] = stale
    return r


def scan_input_list(cfg):
    paths = cfg.get("inputs_tsv")
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]
    out, seen = [], set()
    config_dir = os.path.dirname(os.path.abspath(cfg["_config_path"]))
    for raw_path in paths:
        path = _resolve_path(raw_path, config_dir)
        if not os.path.exists(path):
            raise SystemExit(f"inputs_tsv not found: {path}")
        with open(path) as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if fields[0] == "sample_key":
                    continue
                if len(fields) != 3:
                    raise SystemExit(f"{path}:{n}: expected 3 tab-separated fields: sample_key species h5ad")
                key, sp, h5ad = fields
                if key in seen:
                    raise SystemExit(f"{path}:{n}: duplicate sample_key {key!r}")
                seen.add(key)
                out.append(_stage1_record(cfg, key, sp, _resolve_path(h5ad, os.path.dirname(path))))
    return out


def scan_corpus(cfg):
    """One record per dataset with a usable h5ad, selected by content and flagged
    stale if its stage-1 output is missing or older than the h5ad."""
    listed = scan_input_list(cfg)
    if listed is not None:
        return listed
    out = []
    for key in sorted(os.listdir(cfg["corpus_root"])):
        dsdir = os.path.join(cfg["corpus_root"], key)
        if not os.path.isdir(dsdir):
            continue
        stats = _read_stats(dsdir)
        sp = species_of(dsdir, stats)
        if sp is None:
            continue
        f = choose_h5ad(dsdir, stats)
        if f is None:
            continue
        out.append(_stage1_record(cfg, key, sp, f))
    return out


# ---------------------------------------------------------------------- status
def cmd_status(cfg, args):
    ds = scan_corpus(cfg)
    print(f"corpus_root: {cfg['corpus_root']}")
    print(f"cache_root:  {cfg['cache_root']}")
    for sp in SPECIES:
        s = [d for d in ds if d["species"] == sp]
        stale = [d for d in s if d["stale"]]
        print(f"  {sp:6}: {len(s):3} datasets  |  up-to-date {len(s)-len(stale):3}  |  "
              f"stale/missing {len(stale):3}")
    ds2 = [_stage2_record(cfg, d) for d in ds]
    for sp in SPECIES:
        s = [d for d in ds2 if d["species"] == sp]
        stale = [d for d in s if d["stage2_stale"]]
        print(f"  {sp:6} v2: {len(s):3} datasets  |  up-to-date {len(s)-len(stale):3}  |  "
              f"stale/missing {len(stale):3}")
    tot_stale = [d for d in ds if d["stale"]]
    print(f"  TOTAL : {len(ds)} datasets, {len(tot_stale)} need (re)compute")
    for sp in SPECIES:
        gs = os.path.join(cfg["_dirs"]["master"], f"gene_summary_{sp}.parquet")
        if os.path.exists(gs):
            print(f"  master[{sp}] built: {os.path.getmtime(gs):.0f} "
                  f"({pd.read_parquet(gs).shape[0]} genes)")
    lat = os.path.join(cfg["_dirs"]["vocab"], "latest")
    if os.path.islink(lat) or os.path.exists(lat):
        print(f"  latest vocab -> {os.path.realpath(lat)}")


# --------------------------------------------------------------------- measure
def cmd_measure(cfg, args):
    ds = scan_corpus(cfg)
    todo = ds if args.force else [d for d in ds if d["stale"]]
    if not todo:
        print("nothing to measure (all up-to-date). use --force to recompute.")
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    man = os.path.join(cfg["_dirs"]["jobs"], f"manifest_{stamp}.tsv")
    with open(man, "w") as fh:
        for d in todo:
            fh.write(f"{d['key']}\t{d['species']}\t{d['h5ad']}\t{d['out']}\n")
    sl = cfg["slurm"]
    sb = os.path.join(cfg["_dirs"]["jobs"], f"measure_{stamp}.sbatch")
    with open(sb, "w") as fh:
        fh.write(f"""#!/bin/bash
#SBATCH --job-name=efs-measure
#SBATCH -p {sl['partition']}
#SBATCH --time={sl['time']}
#SBATCH --mem={sl['mem']}
#SBATCH --cpus-per-task={sl['cpus']}
#SBATCH --array=1-{len(todo)}%{sl['array_throttle']}
#SBATCH --output={cfg['_dirs']['logs']}/%A_%a.out
set -euo pipefail
{cfg['venv_python']} {HERE}/worker.py --manifest {man} --index $SLURM_ARRAY_TASK_ID
""")
    res = subprocess.run(["sbatch", sb], capture_output=True, text=True)
    print(res.stdout.strip() or res.stderr.strip())
    jid = res.stdout.strip().split()[-1] if res.returncode == 0 else None
    print(f"submitted array of {len(todo)} task(s) [{cfg['slurm']['array_throttle']} concurrent]")
    return jid


def cmd_measure_v2(cfg, args):
    ds = [_stage2_record(cfg, d) for d in scan_corpus(cfg)]
    todo = ds if args.force else [d for d in ds if d["stage2_stale"]]
    if not todo:
        print("nothing to measure-v2 (all up-to-date). use --force to recompute.")
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    man = os.path.join(cfg["_dirs"]["jobs"], f"manifest_v2_{stamp}.tsv")
    with open(man, "w") as fh:
        for d in todo:
            fh.write(f"{d['key']}\t{d['species']}\t{d['h5ad']}\t{d['stage2_out']}\n")
    sl = cfg.get("slurm_v2", cfg["slurm"])
    sb = os.path.join(cfg["_dirs"]["jobs"], f"measure_v2_{stamp}.sbatch")
    with open(sb, "w") as fh:
        fh.write(f"""#!/bin/bash
#SBATCH --job-name=efs-v2
#SBATCH -p {sl['partition']}
#SBATCH --time={sl['time']}
#SBATCH --mem={sl['mem']}
#SBATCH --cpus-per-task={sl['cpus']}
#SBATCH --array=1-{len(todo)}%{sl['array_throttle']}
#SBATCH --output={cfg['_dirs']['logs']}/%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS={sl['cpus']}
export OPENBLAS_NUM_THREADS={sl['cpus']}
export MKL_NUM_THREADS={sl['cpus']}
{cfg['venv_python']} {HERE}/worker_v2.py --config {cfg['_config_path']} --manifest {man} --index $SLURM_ARRAY_TASK_ID
""")
    res = subprocess.run(["sbatch", sb], capture_output=True, text=True)
    print(res.stdout.strip() or res.stderr.strip())
    jid = res.stdout.strip().split()[-1] if res.returncode == 0 else None
    print(f"submitted v2 array of {len(todo)} task(s) [{sl['array_throttle']} concurrent]")
    return jid


def _wait(jid):
    if not jid:
        return
    print(f"waiting for job {jid} ...")
    while subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True).stdout.strip():
        time.sleep(20)


# ------------------------------------------------------------------- aggregate
def aggregate(cfg):
    records = scan_corpus(cfg)
    species_by_key = {d["key"]: d["species"] for d in records}
    keys = set(species_by_key)  # only datasets currently in the corpus
    files = [os.path.join(cfg["_dirs"]["stage1"], k + ".parquet") for k in keys]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("no stage-1 outputs yet; run `measure` first")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["species"] = df["sample_key"].map(species_by_key).fillna(df["species"])
    df["det"] = df["n_detected"] / df["n_cells"]
    df["mean_all"] = df["sum_counts"] / df["n_cells"]
    df["mean_expr"] = df["sum_counts"] / df["n_detected"].where(df["n_detected"] > 0)
    md = cfg["_dirs"]["master"]
    print(f"aggregating {len(files)} datasets, {len(df)} (dataset,gene) rows")
    for sp, g in df.groupby("species"):
        det = g.pivot_table(index="harmonized_id", columns="sample_key", values="det")
        mean = g.pivot_table(index="harmonized_id", columns="sample_key", values="mean_all")
        det.to_parquet(os.path.join(md, f"det_matrix_{sp}.parquet"))
        mean.to_parquet(os.path.join(md, f"mean_matrix_{sp}.parquet"))
        summary = pd.DataFrame(index=det.index)
        summary["n_datasets_present"] = det.notna().sum(axis=1)
        for t in DET_THRESH:
            summary[f"n_datasets_det_{t}"] = (det >= t).sum(axis=1)
        summary["max_det"] = det.max(axis=1)
        summary["median_det_present"] = det.median(axis=1)
        summary["max_mean_all"] = mean.max(axis=1)
        extra = g.groupby("harmonized_id").agg(
            tot_detected=("n_detected", "sum"), tot_cells_present=("n_cells", "sum"),
            max_mean_expr=("mean_expr", "max"))
        summary = summary.join(extra)
        summary["pooled_det"] = summary["tot_detected"] / summary["tot_cells_present"]
        summary = summary.reset_index()
        ref = os.path.join(cfg["_dirs"]["ref"], f"biotype_{sp}.parquet")
        if os.path.exists(ref):
            summary = summary.merge(pd.read_parquet(ref), on="harmonized_id", how="left")
        summary.to_parquet(os.path.join(md, f"gene_summary_{sp}.parquet"), index=False)
        print(f"  [{sp}] datasets={g['sample_key'].nunique()} genes={det.shape[0]}")


# ---------------------------------------------------------------------- select
def select_snapshot(cfg, f, k, sdet, sq, category_rule, tag):
    md = cfg["_dirs"]["master"]
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(cfg["_dirs"]["vocab"], tag)
    os.makedirs(outdir, exist_ok=True)
    counts = {}
    for sp in SPECIES:
        gs = os.path.join(md, f"gene_summary_{sp}.parquet")
        det_path = os.path.join(md, f"det_matrix_{sp}.parquet")
        if not os.path.exists(gs):
            continue
        s = pd.read_parquet(gs).set_index("harmonized_id")
        det = pd.read_parquet(det_path)
        for c in FLAGCOLS:
            s[c] = s[c].fillna(False).astype(bool) if c in s else False
        if "biotype" not in s:
            s["biotype"] = None
        bt = s["biotype"].fillna("")
        s["is_pseudogene"] = s["is_pseudogene"] | bt.str.contains("pseudogene", case=False, na=False)
        cons = (det >= f).sum(axis=1).reindex(s.index).fillna(0) >= k  # arbitrary f from matrix
        qbar = s.loc[cons, "max_mean_expr"].quantile(sq) if cons.any() else np.inf
        strength = (s["max_det"] >= sdet) & (s["max_mean_expr"] >= qbar)
        s["candidate"] = cons | strength
        cand = s[s["candidate"]].copy()
        bt = cand["biotype"].fillna("")
        cand["veto_narrow"] = cand[NARROW].any(axis=1)
        cand["veto_wide"] = cand["veto_narrow"] | (bt.ne("protein_coding") & bt.ne(""))
        unknown = [c for c in category_rule if c not in cand.columns]
        if unknown:
            raise SystemExit(f"unknown category_rule flag(s): {unknown}")
        applied = cand[category_rule].any(axis=1) if category_rule else pd.Series(False, index=cand.index)
        cand["category_excluded"] = applied
        cand["selected"] = ~applied
        cand.index.name = "harmonized_id"
        cols = (["symbol", "biotype", "selected", "category_excluded", "veto_narrow", "veto_wide",
                 "n_datasets_present", "max_det", "median_det_present",
                 "max_mean_expr", "pooled_det"] + FLAGCOLS)
        cols = [c for c in cols if c in cand.columns]
        cand[cols].sort_values(["selected", "n_datasets_present", "max_det"],
                               ascending=False).to_csv(
            os.path.join(outdir, f"vocab_{sp}.tsv"), sep="\t")
        counts[sp] = {"candidate": int(len(cand)), "selected": int(cand["selected"].sum())}
        print(f"  [{sp}] candidate={counts[sp]['candidate']} selected={counts[sp]['selected']}")
    meta = {"tag": tag, "f": f, "k": k, "strength_det": sdet, "strength_q": sq,
            "category_rule": category_rule, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "counts": counts}
    json.dump(meta, open(os.path.join(outdir, "params.json"), "w"), indent=2)
    latest = os.path.join(cfg["_dirs"]["vocab"], "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(tag, latest)
    print(f"-> snapshot {outdir}  (latest -> {tag})")


def _select_params(cfg, args):
    d = cfg["select"]
    category_rule = d.get("category_rule", d.get("veto", []))
    if args.veto:
        category_rule = args.veto.split(",")
    if args.category_rule:
        category_rule = args.category_rule.split(",")
    return (args.f if args.f is not None else d["f"],
            args.k if args.k is not None else d["k"],
            d["strength_det"], d["strength_q"],
            category_rule)


def cmd_build(cfg, args):
    aggregate(cfg)
    f, k, sdet, sq, category_rule = _select_params(cfg, args)
    print(f"selecting (f={f}, k={k}, strength={sdet}/q{sq}, category_rule={category_rule or 'none'})")
    select_snapshot(cfg, f, k, sdet, sq, category_rule, args.tag)


def cmd_refresh(cfg, args):
    jid = cmd_measure(cfg, args)
    _wait(jid)
    cmd_build(cfg, args)


# --------------------------------------------------------------------- v2 build
def _select_v2_params(cfg, args):
    d = cfg["select_v2"]
    det = d["detection_rule"]
    hvg = d["hvg_rule"]
    deg = d["cluster_deg_rule"]
    category_rule = d.get("category_rule", [])
    if getattr(args, "category_rule", None):
        category_rule = args.category_rule.split(",")
    if getattr(args, "veto", None):
        category_rule = args.veto.split(",")
    return {
        "min_frac": args.min_frac if getattr(args, "min_frac", None) is not None else det["min_frac"],
        "min_dataset_occurrence": (
            args.min_dataset_occurrence if getattr(args, "min_dataset_occurrence", None) is not None
            else det["min_dataset_occurrence"]
        ),
        "hvg_min_datasets": (
            args.hvg_min_datasets if getattr(args, "hvg_min_datasets", None) is not None
            else hvg.get("min_datasets", 1)
        ),
        "deg_min_datasets": (
            args.deg_min_datasets if getattr(args, "deg_min_datasets", None) is not None
            else deg.get("min_datasets", 1)
        ),
        "category_rule": category_rule,
    }


def _read_stage2_current(cfg):
    rows = []
    records = [_stage2_record(cfg, d) for d in scan_corpus(cfg)]
    missing, stale = 0, 0
    cols = ["sample_key", "species", "harmonized_id", "selected_hvg", "selected_cluster_deg"]
    for rec in records:
        path = rec["stage2_out"]
        if not os.path.exists(path):
            missing += 1
            continue
        if rec["stage2_stale"]:
            stale += 1
        df = pd.read_parquet(path)
        for c in cols:
            if c not in df:
                df[c] = False if c.startswith("selected_") else None
        rows.append(df)
    if missing or stale:
        print(f"warning: build-v2 using stage2 cache with {stale} stale and {missing} missing dataset(s)")
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True)


def _support_counts(stage2, flag):
    if stage2.empty or flag not in stage2:
        return pd.Series(dtype=np.int64)
    x = stage2[stage2[flag].fillna(False)]
    if x.empty:
        return pd.Series(dtype=np.int64)
    return x.drop_duplicates(["sample_key", "harmonized_id"]).groupby("harmonized_id")["sample_key"].nunique()


def _add_ref_annotations(cfg, sp, frame):
    ref = os.path.join(cfg["_dirs"]["ref"], f"biotype_{sp}.parquet")
    if not os.path.exists(ref):
        return frame
    r = pd.read_parquet(ref).set_index("harmonized_id")
    for col in r.columns:
        if col not in frame:
            frame[col] = r[col].reindex(frame.index)
        else:
            frame[col] = frame[col].combine_first(r[col].reindex(frame.index))
    return frame


def _prepare_category_flags(frame):
    for c in FLAGCOLS:
        frame[c] = frame[c].fillna(False).astype(bool) if c in frame else False
    if "biotype" not in frame:
        frame["biotype"] = None
    bt = frame["biotype"].fillna("")
    frame["is_pseudogene"] = frame["is_pseudogene"] | bt.str.contains("pseudogene", case=False, na=False)
    frame["veto_narrow"] = frame[NARROW].any(axis=1)
    frame["veto_wide"] = frame["veto_narrow"] | (bt.ne("protein_coding") & bt.ne(""))
    return frame


def select_snapshot_v2(cfg, params, tag):
    aggregate(cfg)
    md = cfg["_dirs"]["master"]
    stage2 = _read_stage2_current(cfg)
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(cfg["_dirs"]["vocab"], tag)
    os.makedirs(outdir, exist_ok=True)
    counts = {}
    for sp in SPECIES:
        gs = os.path.join(md, f"gene_summary_{sp}.parquet")
        det_path = os.path.join(md, f"det_matrix_{sp}.parquet")
        if not os.path.exists(gs) or not os.path.exists(det_path):
            continue
        summary = pd.read_parquet(gs).set_index("harmonized_id")
        det = pd.read_parquet(det_path)
        sp_stage2 = stage2[stage2["species"] == sp].copy()
        hvg_counts = _support_counts(sp_stage2, "selected_hvg")
        deg_counts = _support_counts(sp_stage2, "selected_cluster_deg")

        n_det = (det >= params["min_frac"]).sum(axis=1)
        idx = summary.index.union(hvg_counts.index).union(deg_counts.index).union(n_det.index)
        s = summary.reindex(idx)
        s = _add_ref_annotations(cfg, sp, s)
        s = _prepare_category_flags(s)

        s["n_datasets_detection"] = n_det.reindex(idx).fillna(0).astype(int)
        s["n_datasets_hvg"] = hvg_counts.reindex(idx).fillna(0).astype(int)
        s["n_datasets_cluster_deg"] = deg_counts.reindex(idx).fillna(0).astype(int)
        s["selected_by_detection"] = s["n_datasets_detection"] >= params["min_dataset_occurrence"]
        s["selected_by_hvg"] = s["n_datasets_hvg"] >= params["hvg_min_datasets"]
        s["selected_by_cluster_deg"] = s["n_datasets_cluster_deg"] >= params["deg_min_datasets"]
        s["candidate"] = s["selected_by_detection"] | s["selected_by_hvg"] | s["selected_by_cluster_deg"]

        cand = s[s["candidate"]].copy()
        unknown = [c for c in params["category_rule"] if c not in cand.columns]
        if unknown:
            raise SystemExit(f"unknown category_rule flag(s): {unknown}")
        applied = cand[params["category_rule"]].any(axis=1) if params["category_rule"] else pd.Series(False, index=cand.index)
        cand["category_excluded"] = applied
        cand["selected"] = ~applied
        cand.index.name = "harmonized_id"

        cols = (["symbol", "biotype", "selected", "category_excluded",
                 "selected_by_detection", "selected_by_hvg", "selected_by_cluster_deg",
                 "n_datasets_detection", "n_datasets_hvg", "n_datasets_cluster_deg",
                 "n_datasets_present", "max_det", "median_det_present", "pooled_det",
                 "veto_narrow", "veto_wide"] + FLAGCOLS)
        cols = [c for c in cols if c in cand.columns]
        cand[cols].sort_values(
            ["selected", "selected_by_detection", "n_datasets_detection",
             "n_datasets_hvg", "n_datasets_cluster_deg"],
            ascending=False,
        ).to_csv(os.path.join(outdir, f"vocab_{sp}.tsv"), sep="\t")

        counts[sp] = {
            "candidate": int(len(cand)),
            "selected": int(cand["selected"].sum()),
            "detection": int(cand["selected_by_detection"].sum()),
            "hvg": int(cand["selected_by_hvg"].sum()),
            "cluster_deg": int(cand["selected_by_cluster_deg"].sum()),
        }
        print(f"  [{sp}] candidate={counts[sp]['candidate']} selected={counts[sp]['selected']} "
              f"detection={counts[sp]['detection']} hvg={counts[sp]['hvg']} "
              f"cluster_deg={counts[sp]['cluster_deg']}")

    meta = {
        "policy": "v2",
        "tag": tag,
        "select_v2": params,
        "selection_v2_measure_signature": selection_v2_measure_signature(cfg),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": counts,
    }
    json.dump(meta, open(os.path.join(outdir, "params.json"), "w"), indent=2)
    latest = os.path.join(cfg["_dirs"]["vocab"], "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(tag, latest)
    print(f"-> snapshot {outdir}  (latest -> {tag})")


def cmd_build_v2(cfg, args):
    params = _select_v2_params(cfg, args)
    print("selecting v2 "
          f"(min_frac={params['min_frac']}, min_dataset_occurrence={params['min_dataset_occurrence']}, "
          f"hvg_min_datasets={params['hvg_min_datasets']}, "
          f"deg_min_datasets={params['deg_min_datasets']}, "
          f"category_rule={params['category_rule'] or 'none'})")
    select_snapshot_v2(cfg, params, args.tag)


def cmd_refresh_v2(cfg, args):
    jid1 = cmd_measure(cfg, args)
    jid2 = cmd_measure_v2(cfg, args)
    _wait(jid1)
    _wait(jid2)
    cmd_build_v2(cfg, args)


# ------------------------------------------------------------------------ diff
def _load_vocab(cfg, tag, sp):
    p = os.path.join(cfg["_dirs"]["vocab"], tag, f"vocab_{sp}.tsv")
    d = pd.read_csv(p, sep="\t", index_col=0)
    return d[d["selected"]] if "selected" in d else d


def cmd_diff(cfg, args):
    for sp in SPECIES:
        try:
            a = _load_vocab(cfg, args.a, sp)
            b = _load_vocab(cfg, args.b, sp)
        except FileNotFoundError:
            continue
        A, B = set(a.index), set(b.index)
        added, removed = B - A, A - B
        print(f"\n[{sp}] {args.a}({len(A)}) -> {args.b}({len(B)})  "
              f"+{len(added)} / -{len(removed)}")
        for label, ids in [("added", added), ("removed", removed)]:
            ref = b if label == "added" else a
            ex = [f"{i}({ref.loc[i, 'symbol']})" for i in list(ids)[:8] if i in ref.index]
            if ex:
                print(f"    {label}: " + ", ".join(ex) + (" ..." if len(ids) > 8 else ""))


def cmd_list(cfg, args):
    vd = cfg["_dirs"]["vocab"]
    for tag in sorted(os.listdir(vd)):
        p = os.path.join(vd, tag, "params.json")
        if os.path.exists(p):
            m = json.load(open(p))
            c = m.get("counts", {})
            sel = {sp: c[sp]["selected"] for sp in c}
            if m.get("policy") == "v2":
                v2 = m.get("select_v2", {})
                print(f"  {tag}  v2 min_frac={v2.get('min_frac')} "
                      f"min_dataset_occurrence={v2.get('min_dataset_occurrence')} "
                      f"category_rule={v2.get('category_rule') or 'none'}  selected={sel}")
            else:
                category_rule = m.get("category_rule", m.get("veto", []))
                print(f"  {tag}  f={m['f']} k={m['k']} category_rule={category_rule or 'none'}  selected={sel}")


# ------------------------------------------------------------ biotype reference
_ATTR = re.compile(r'(\w+) "([^"]*)"')
SEX = {"human": {"XIST", "TSIX", "RPS4Y1", "RPS4Y2", "DDX3Y", "UTY", "USP9Y", "KDM5D",
                 "EIF1AY", "NLGN4Y", "ZFY", "TXLNGY", "PRKY", "TMSB4Y"},
       "mouse": {"Xist", "Tsix", "Ddx3y", "Uty", "Eif2s3y", "Kdm5d", "Zfy1", "Zfy2", "Uba1y"}}


def _parse_gtf(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fld = line.split("\t", 8)
            if len(fld) < 9 or fld[2] != "gene":
                continue
            a = dict(_ATTR.findall(fld[8]))
            gid = a.get("gene_id", "")
            if gid:
                rows.append((gid.split(".")[0], a.get("gene_name", ""), a.get("gene_biotype", "")))
    return pd.DataFrame(rows, columns=["harmonized_id", "symbol", "biotype"]).drop_duplicates("harmonized_id")


def _add_flags(df, sp):
    s, bt = df["symbol"].fillna(""), df["biotype"].fillna("")
    df["is_pseudogene"] = bt.str.contains("pseudogene", case=False, na=False)
    if sp == "human":
        df["is_OR"] = s.str.match(r"^OR\d")
        df["is_vomeronasal"] = s.str.match(r"^VN[12]R")
        df["is_taste"] = s.str.match(r"^TAS[12]R")
        df["is_mt"] = s.str.match(r"^MT-")
        df["is_hb"] = s.str.match(r"^HB[ABDEGMQZ]\d?$") | s.isin({"HBB", "HBA1", "HBA2", "HBD"})
        df["is_ribo"] = s.str.match(r"^RP[SL]\d")
    else:
        df["is_OR"] = s.str.match(r"^Or\d") | s.str.match(r"^Olfr")  # GRCm39: Olfr*->Or<digit>
        df["is_vomeronasal"] = s.str.match(r"^Vmn[12]r")
        df["is_taste"] = s.str.match(r"^Tas[12]r")
        df["is_mt"] = s.str.match(r"^mt-")
        df["is_hb"] = s.str.match(r"^Hb[abq]")
        df["is_ribo"] = s.str.match(r"^Rp[sl]\d")
    df["is_sex"] = s.isin(SEX[sp])
    for seg in ["V", "D", "J", "C"]:
        df[f"is_IG_{seg}"] = bt == f"IG_{seg}_gene"
        df[f"is_TR_{seg}"] = bt == f"TR_{seg}_gene"
    return df


def cmd_ref(cfg, args):
    rel = cfg["ensembl_release"]
    urls = {
        "human": f"https://ftp.ensembl.org/pub/release-{rel}/gtf/homo_sapiens/Homo_sapiens.GRCh38.{rel}.gtf.gz",
        "mouse": f"https://ftp.ensembl.org/pub/release-{rel}/gtf/mus_musculus/Mus_musculus.GRCm39.{rel}.gtf.gz",
    }
    for sp, url in urls.items():
        gz = os.path.join(cfg["_dirs"]["ref"], os.path.basename(url))
        out = os.path.join(cfg["_dirs"]["ref"], f"biotype_{sp}.parquet")
        if os.path.exists(out) and not args.force:
            print(f"[{sp}] exists (use --force): {out}")
            continue
        if not os.path.exists(gz):
            print(f"[{sp}] downloading {url}")
            urllib.request.urlretrieve(url, gz)
        df = _add_flags(_parse_gtf(gz), sp)
        df.to_parquet(out, index=False)
        print(f"[{sp}] genes={len(df)} protein_coding="
              f"{int((df['biotype'] == 'protein_coding').sum())} -> {out}")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(prog="featuresel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("measure"); m.add_argument("--force", action="store_true")
    mv2 = sub.add_parser("measure-v2"); mv2.add_argument("--force", action="store_true")
    r = sub.add_parser("ref"); r.add_argument("--force", action="store_true")

    def add_build_opts(p):
        p.add_argument("--f", type=float); p.add_argument("--k", type=int)
        p.add_argument("--category-rule")
        p.add_argument("--veto", help=argparse.SUPPRESS)  # backward-compatible alias
        p.add_argument("--tag")
        p.add_argument("--force", action="store_true")
    add_build_opts(sub.add_parser("build"))
    add_build_opts(sub.add_parser("refresh"))
    def add_build_v2_opts(p):
        p.add_argument("--min-frac", type=float)
        p.add_argument("--min-dataset-occurrence", type=int)
        p.add_argument("--hvg-min-datasets", type=int)
        p.add_argument("--deg-min-datasets", type=int)
        p.add_argument("--category-rule")
        p.add_argument("--veto", help=argparse.SUPPRESS)
        p.add_argument("--tag")
        p.add_argument("--force", action="store_true")
    add_build_v2_opts(sub.add_parser("build-v2"))
    add_build_v2_opts(sub.add_parser("refresh-v2"))
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b")
    sub.add_parser("list")

    args = ap.parse_args()
    cfg = load_config(args.config)
    {"status": cmd_status, "measure": cmd_measure, "measure-v2": cmd_measure_v2,
     "ref": cmd_ref, "build": cmd_build, "build-v2": cmd_build_v2,
     "refresh": cmd_refresh, "refresh-v2": cmd_refresh_v2,
     "diff": cmd_diff, "list": cmd_list}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
