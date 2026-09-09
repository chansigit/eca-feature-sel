#!/usr/bin/env python
"""eca-feature-sel: HVG-union gene vocabulary for single-cell FMs.

Per dataset: top-N HVGs + basic expression stats (Slurm-parallel, cached).
Across datasets: union the HVGs, then apply gene-category keep/drop lists.
Human and mouse are built independently in their own Ensembl ID spaces.

Subcommands:
  status              inputs vs cache: done / stale / missing
  measure [--force]   submit one array task per dataset (--local runs here)
  ref [--force]       build the Ensembl biotype/flag reference (needs internet)
  build [opts]        union cached HVGs + category rules -> vocab snapshot
  refresh             measure -> wait -> build
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd
import yaml

pd.set_option("future.no_silent_downcasting", True)  # quiet fillna(False) on flag cols

HERE = os.path.dirname(os.path.abspath(__file__))
SPECIES = ("human", "mouse")
HVG_SOURCES = ["union", "global", "lineage", "upstream"]
FLAGCOLS = ["is_protein_coding", "is_pseudogene", "is_OR", "is_vomeronasal", "is_taste",
            "is_IG_V", "is_IG_D", "is_IG_J", "is_IG_C", "is_TR_V", "is_TR_D", "is_TR_J",
            "is_TR_C", "is_mt", "is_hb", "is_ribo", "is_sex"]


# ---------------------------------------------------------------- config / inputs
def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    c = cfg["cache_root"]
    cfg["_dirs"] = {d: os.path.join(c, d) for d in ("stats", "ref", "vocab", "jobs")}
    cfg["_dirs"]["logs"] = os.path.join(c, "jobs", "logs")
    for d in cfg["_dirs"].values():
        os.makedirs(d, exist_ok=True)
    return cfg


def hvg_signature(cfg):
    """Cached per-dataset results are stale when the HVG settings change."""
    txt = json.dumps(cfg["hvg"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode()).hexdigest()


def _resolve(path, base):
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))


def scan_inputs(cfg):
    """One record per dataset from inputs_tsv (sample_key, species, h5ad)."""
    paths = cfg["inputs_tsv"]
    if isinstance(paths, str):
        paths = [paths]
    sig = hvg_signature(cfg)
    out, seen = [], set()
    for raw in paths:
        path = _resolve(raw, os.path.dirname(cfg["_config_path"]))
        if not os.path.exists(path):
            raise SystemExit(f"inputs_tsv not found: {path}")
        for n, line in enumerate(open(path), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] == "sample_key":
                continue
            if len(fields) != 3:
                raise SystemExit(f"{path}:{n}: expected 3 tab-separated fields")
            key, sp, h5ad = fields
            if sp not in SPECIES:
                raise SystemExit(f"{path}:{n}: species={sp!r} (expected one of {SPECIES})")
            if key in seen:
                raise SystemExit(f"{path}:{n}: duplicate sample_key {key!r}")
            seen.add(key)
            h5ad = _resolve(h5ad, os.path.dirname(path))
            if not os.path.exists(h5ad):
                raise SystemExit(f"{path}:{n}: h5ad not found: {h5ad}")
            out.append({"key": key, "species": sp, "h5ad": h5ad,
                        "out": os.path.join(cfg["_dirs"]["stats"], key + ".parquet")})
    for r in out:
        r["stale"] = _is_stale(r, sig)
    return out


def _is_stale(rec, sig):
    if not os.path.exists(rec["out"]):
        return True
    if os.path.getmtime(rec["h5ad"]) > os.path.getmtime(rec["out"]):
        return True
    meta = os.path.splitext(rec["out"])[0] + ".meta.json"
    try:
        return json.load(open(meta)).get("hvg_signature") != sig
    except Exception:
        return True


# ---------------------------------------------------------------- status/measure
def cmd_status(cfg, args):
    ds = scan_inputs(cfg)
    print(f"cache_root: {cfg['cache_root']}")
    for sp in SPECIES:
        s = [d for d in ds if d["species"] == sp]
        stale = [d for d in s if d["stale"]]
        print(f"  {sp:6}: {len(s):3} datasets | up-to-date {len(s) - len(stale):3} | "
              f"stale/missing {len(stale):3}")
    print(f"  TOTAL: {len(ds)} datasets, {sum(d['stale'] for d in ds)} need (re)compute")
    for sp in SPECIES:
        v = os.path.join(cfg["_dirs"]["vocab"], "latest", f"vocab_{sp}.tsv")
        if os.path.exists(v):
            print(f"  latest vocab[{sp}]: {sum(1 for _ in open(v)) - 1} rows")


def cmd_measure(cfg, args):
    ds = scan_inputs(cfg)
    todo = ds if args.force else [d for d in ds if d["stale"]]
    if not todo:
        print("nothing to measure (all up-to-date). use --force to recompute.")
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    man = os.path.join(cfg["_dirs"]["jobs"], f"manifest_{stamp}.tsv")
    with open(man, "w") as fh:
        for d in todo:
            fh.write(f"{d['key']}\t{d['species']}\t{d['h5ad']}\t{d['out']}\n")

    q = shlex.quote
    worker = f"{q(cfg['venv_python'])} {q(os.path.join(HERE, 'worker.py'))} --config {q(cfg['_config_path'])}"
    if args.local:
        jobs = args.jobs or max(1, len(os.sched_getaffinity(0)))
        print(f"running {len(todo)} dataset(s) locally with {jobs} process(es)", flush=True)
        subprocess.run(f"{worker} --manifest {q(man)} --jobs {jobs}", shell=True, check=True)
        return None
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
export OMP_NUM_THREADS={sl['cpus']} OPENBLAS_NUM_THREADS={sl['cpus']} MKL_NUM_THREADS={sl['cpus']}
{worker} --manifest {q(man)} --index $SLURM_ARRAY_TASK_ID
""")
    res = subprocess.run(["sbatch", sb], capture_output=True, text=True)
    print(res.stdout.strip() or res.stderr.strip())
    print(f"submitted array of {len(todo)} task(s) [{sl['array_throttle']} concurrent]")
    return res.stdout.strip().split()[-1] if res.returncode == 0 else None


def _wait(jid):
    if not jid:
        return
    print(f"waiting for job {jid} ...")
    while subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True).stdout.strip():
        time.sleep(20)


# ----------------------------------------------------------------------- build
def _load_stats(cfg, ds):
    """Cached per-dataset gene stats + per-(lineage, gene) HVG detail."""
    frames, lineages, missing, stale = [], [], 0, 0
    for rec in ds:
        if not os.path.exists(rec["out"]):
            missing += 1
            continue
        stale += bool(rec["stale"])
        frames.append(pd.read_parquet(rec["out"]))
        detail = os.path.splitext(rec["out"])[0] + ".lineages.parquet"
        if os.path.exists(detail):
            d = pd.read_parquet(detail)
            d["species"] = rec["species"]
            lineages.append(d)
    if missing or stale:
        print(f"warning: building with {stale} stale and {missing} missing dataset(s)")
    if not frames:
        raise SystemExit("no measurements yet; run `measure` first")
    lin = pd.concat(lineages, ignore_index=True) if lineages else pd.DataFrame(
        columns=["sample_key", "lineage", "lineage_cells", "harmonized_id", "species"])
    return pd.concat(frames, ignore_index=True), lin


def _passes(df, s_min, n_max):
    """The selection rule, applied to any table with score + rank columns."""
    return (df["rank"] > 0) & (df["rank"] <= n_max) & (df["score"] >= s_min)


def _lineage_votes(lin, s_min, n_max, min_cells, min_lineages):
    """(sample_key, harmonized_id) pairs where the gene passes in >= min_lineages
    lineages of at least min_cells cells."""
    cols = ["sample_key", "harmonized_id"]
    if lin.empty:
        return pd.DataFrame(columns=cols)
    kept = lin[(lin["lineage_cells"] >= min_cells) & _passes(lin, s_min, n_max)]
    per = kept.groupby(cols)["lineage"].nunique()
    return per[per >= min_lineages].reset_index()[cols]


def _lineage_support(lin, s_min, n_max, min_cells, min_lineages):
    """Datasets per gene that vote via their lineages."""
    v = _lineage_votes(lin, s_min, n_max, min_cells, min_lineages)
    return v.groupby("harmonized_id")["sample_key"].nunique() if len(v) else pd.Series(dtype=np.int64)


def _gene_table(df, s_min, n_max):
    """Per-gene stats across datasets: HVG support + how highly it is expressed."""
    df = df.copy()
    df["det"] = df["n_detected"] / df["n_cells"]
    # fillna before grouping: SeriesGroupBy.fillna returns a row-aligned Series,
    # whose .sum() is a scalar that would silently broadcast over every gene.
    df["hvg_upstream"] = (df["hvg_upstream"].fillna(False).astype(bool)
                          if "hvg_upstream" in df else False)
    df["hvg"] = _passes(df, s_min, n_max)
    g = df.groupby("harmonized_id")
    t = pd.DataFrame({
        "n_datasets_present": g["sample_key"].nunique(),
        "n_datasets_hvg_global": g["hvg"].sum().astype(int),
        "n_datasets_hvg_upstream": g["hvg_upstream"].sum().astype(int),
        "best_score": g["score"].max(),
        "median_score": g["score"].median(),
        "tot_detected": g["n_detected"].sum(),
        "tot_cells": g["n_cells"].sum(),
        "tot_counts": g["sum_counts"].sum(),
        "max_det": g["det"].max(),
        "median_det": g["det"].median(),
    })
    t["pooled_det"] = t["tot_detected"] / t["tot_cells"]
    t["mean_counts_per_cell"] = t["tot_counts"] / t["tot_cells"]
    t["mean_counts_in_positive"] = t["tot_counts"] / t["tot_detected"].where(t["tot_detected"] > 0)
    return t.drop(columns=["tot_detected", "tot_counts"])


def _union_support(sub, lin, s_min, n_max, min_cells, min_lineages):
    """Datasets per gene voting through the dataset-level call OR their lineages."""
    glob = sub[_passes(sub, s_min, n_max)][["sample_key", "harmonized_id"]]
    via_lin = _lineage_votes(lin, s_min, n_max, min_cells, min_lineages)
    both = pd.concat([glob, via_lin], ignore_index=True).drop_duplicates()
    return both.groupby("harmonized_id")["sample_key"].nunique()


def _annotate(cfg, sp, t):
    ref = os.path.join(cfg["_dirs"]["ref"], f"biotype_{sp}.parquet")
    if os.path.exists(ref):
        t = t.join(pd.read_parquet(ref).set_index("harmonized_id"), how="left")
    if "biotype" not in t:
        t["biotype"] = None
    for c in FLAGCOLS:
        t[c] = t[c].fillna(False).astype(bool) if c in t else False
    return t


def _flag_any(t, keys):
    """Genes in any of the categories; a key is an is_* flag or an Ensembl biotype."""
    seen = set(t["biotype"].dropna()) if "biotype" in t else set()
    for k in keys:
        if k not in FLAGCOLS and k not in seen:
            print(f"    warning: category key {k!r} matches no gene "
                  f"(not an is_* flag, not a biotype in this corpus)")
    m = pd.Series(False, index=t.index)
    for k in keys:
        m |= _category_mask(t, k)
    return m


def _category_mask(t, key):
    """A rule key is either an is_* flag column or an exact Ensembl biotype."""
    if key in FLAGCOLS:
        return t[key].astype(bool)
    return t["biotype"].fillna("") == key


def _min_datasets(t, rules):
    """Per-gene HVG-support threshold. First matching rule wins (config order),
    so put narrow categories above broad ones; `default` covers the rest."""
    out = pd.Series(int(rules.get("default", 1)), index=t.index, dtype=int)
    taken = pd.Series(False, index=t.index)
    seen_biotypes = set(t["biotype"].dropna())
    for key, val in rules.items():
        if key == "default":
            continue
        if key not in FLAGCOLS and key not in seen_biotypes:
            print(f"    warning: rule key {key!r} matches no gene "
                  f"(not an is_* flag, not a biotype in this corpus)")
        m = _category_mask(t, key) & ~taken
        out[m] = int(val)
        taken |= m
    return out


def _threshold_table(t, ks=(1, 2, 3, 5, 10, 20, 50)):
    """How many genes per category survive at each candidate threshold."""
    rows = []
    for label, sub in [("ALL", t)] + [(b, g) for b, g in t.groupby(t["biotype"].fillna("(none)"))]:
        row = {"category": label, "genes_seen": len(sub)}
        row.update({f"hvg_ge_{k}": int((sub["n_datasets_hvg"] >= k).sum()) for k in ks})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("genes_seen", ascending=False)


def _sweep_table(sub, lin, rules, min_cells, min_lineages, ref_biotype,
                 s_mins=(1.1, 1.2, 1.3, 1.5, 2.0), n_max=3000):
    """How many genes the category rules would select at each s_min, and how the
    per-dataset vote size moves with it. Answers "is s_min too strict?" at the
    corpus level without remeasuring."""
    rows = []
    for s_min in s_mins:
        votes = _union_support(sub, lin, s_min, n_max, min_cells, min_lineages)
        per_ds = pd.concat([sub[_passes(sub, s_min, n_max)][["sample_key", "harmonized_id"]],
                            _lineage_votes(lin, s_min, n_max, min_cells, min_lineages)]
                           ).drop_duplicates().groupby("sample_key").size()
        t = pd.DataFrame({"n_datasets_hvg": votes})
        t["biotype"] = ref_biotype.reindex(t.index) if ref_biotype is not None else None
        req = _min_datasets(_with_flags(t), rules)
        sel = t["n_datasets_hvg"] >= req
        rows.append({"s_min": s_min, "votes_per_dataset_median": int(per_ds.median()) if len(per_ds) else 0,
                     "votes_per_dataset_min": int(per_ds.min()) if len(per_ds) else 0,
                     "votes_per_dataset_max": int(per_ds.max()) if len(per_ds) else 0,
                     "genes_with_any_vote": int(len(t)),
                     "selected_by_category_rules": int(sel.sum()),
                     "selected_protein_coding": int((sel & (t["biotype"] == "protein_coding")).sum()),
                     "selected_lncRNA": int((sel & (t["biotype"] == "lncRNA")).sum())})
    return pd.DataFrame(rows)


def _with_flags(t):
    for c in FLAGCOLS:
        if c not in t:
            t[c] = False
    return t


def _parse_min_rules(spec, base):
    """--hvg-min-datasets accepts a bare int (flat) or `key=n,key=n` overrides."""
    if spec is None:
        return dict(base)
    if spec.strip().lstrip("-").isdigit():
        return {"default": int(spec)}
    rules = dict(base)
    for part in spec.split(","):
        key, _, val = part.partition("=")
        if not val:
            raise SystemExit(f"--hvg-min-datasets: expected key=n, got {part!r}")
        rules[key.strip()] = int(val)
    return rules


def cmd_build(cfg, args):
    sel = cfg["selection"]
    base = sel.get("hvg_min_datasets", 1)
    base = {"default": int(base)} if isinstance(base, int) else dict(base or {})
    rules = _parse_min_rules(args.hvg_min_datasets, base)
    keep = (args.category_keep.split(",") if args.category_keep is not None
            else sel.get("category_keep", []) or [])
    drop = (args.category_drop.split(",") if args.category_drop is not None
            else sel.get("category_drop", []) or [])
    source = args.hvg_source or sel.get("hvg_source", "union")
    if source not in HVG_SOURCES:
        raise SystemExit(f"--hvg-source must be one of {HVG_SOURCES}")
    min_cells = (args.min_lineage_cells if args.min_lineage_cells is not None
                 else sel.get("min_lineage_cells", 200))
    min_lineages = (args.min_lineages if args.min_lineages is not None
                    else sel.get("min_lineages", 1))
    s_min = args.s_min if args.s_min is not None else float(sel.get("s_min", 1.5))
    n_max = args.n_max if args.n_max is not None else int(sel.get("n_max", 3000))
    print(f"selecting (hvg_source={source}, s_min={s_min}, n_max={n_max}, "
          f"min_lineage_cells={min_cells}, min_lineages={min_lineages}, "
          f"hvg_min_datasets={rules}, category_keep={keep or 'none'}, category_drop={drop or 'none'})")

    ds = scan_inputs(cfg)
    stats, lin = _load_stats(cfg, ds)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(cfg["_dirs"]["vocab"], tag)
    os.makedirs(outdir, exist_ok=True)
    counts = {}
    for sp in SPECIES:
        sub = stats[stats["species"] == sp]
        if sub.empty:
            continue
        t = _annotate(cfg, sp, _gene_table(sub, s_min, n_max))
        lin_sp = lin[lin["species"] == sp]
        lin_sup = _lineage_support(lin_sp, s_min, n_max, min_cells, min_lineages)
        t["n_datasets_hvg_lineage"] = lin_sup.reindex(t.index).fillna(0).astype(int)
        if source == "union":   # a dataset votes via its global call OR its lineages
            t["n_datasets_hvg"] = _union_support(sub, lin_sp, s_min, n_max, min_cells,
                                                 min_lineages).reindex(t.index).fillna(0).astype(int)
        else:
            t["n_datasets_hvg"] = t[f"n_datasets_hvg_{source}"]
        t["min_datasets_required"] = _min_datasets(t, rules)
        t["hvg_union"] = t["n_datasets_hvg"] >= t["min_datasets_required"]
        t["category_kept"] = _flag_any(t, keep)     # force-in, overrides drop
        t["category_dropped"] = _flag_any(t, drop) & ~t["category_kept"]
        t["selected"] = (t["hvg_union"] | t["category_kept"]) & ~t["category_dropped"]
        t.index.name = "harmonized_id"

        cols = (["symbol", "biotype", "selected", "hvg_union", "category_kept",
                 "category_dropped", "n_datasets_hvg", "min_datasets_required",
                 "n_datasets_hvg_global", "n_datasets_hvg_lineage",
                 "n_datasets_hvg_upstream", "best_score", "median_score", "n_datasets_present",
                 "pooled_det", "max_det", "median_det", "mean_counts_per_cell",
                 "mean_counts_in_positive"] + FLAGCOLS)
        t[[c for c in cols if c in t]].sort_values(
            ["selected", "n_datasets_hvg", "pooled_det"], ascending=False
        ).to_csv(os.path.join(outdir, f"vocab_{sp}.tsv"), sep="\t")
        # the deliverable: just the selected genes
        t.loc[t["selected"], ["symbol", "biotype", "n_datasets_hvg"]].sort_values(
            "n_datasets_hvg", ascending=False
        ).to_csv(os.path.join(outdir, f"genes_{sp}.tsv"), sep="\t")
        # tuning aids: gene counts per biotype at a range of dataset thresholds, and
        # the whole selection re-run at a range of s_min
        _threshold_table(t).to_csv(os.path.join(outdir, f"thresholds_{sp}.tsv"),
                                   sep="\t", index=False)
        ref_bt = t["biotype"] if "biotype" in t else None
        sweep = _sweep_table(sub, lin_sp, rules, min_cells, min_lineages, ref_bt, n_max=n_max)
        sweep.to_csv(os.path.join(outdir, f"sweep_smin_{sp}.tsv"), sep="\t", index=False)
        print("      s_min sweep (votes/dataset median, selected):",
              "  ".join(f"{r.s_min}: {r.votes_per_dataset_median}/{r.selected_by_category_rules}"
                        for r in sweep.itertuples()))

        counts[sp] = {"genes_seen": int(len(t)), "hvg_union": int(t["hvg_union"].sum()),
                      "selected": int(t["selected"].sum()),
                      "datasets": int(sub["sample_key"].nunique())}
        print(f"  [{sp}] datasets={counts[sp]['datasets']} genes_seen={counts[sp]['genes_seen']} "
              f"hvg_union={counts[sp]['hvg_union']} selected={counts[sp]['selected']}")
        by_bt = t[t["selected"]].groupby(t["biotype"].fillna("(none)")).size().sort_values(ascending=False)
        for bt, n in by_bt.head(8).items():
            print(f"      {bt:<28} {n:6}")

    json.dump({"tag": tag, "hvg": cfg["hvg"], "hvg_source": source,
               "s_min": s_min, "n_max": n_max,
               "min_lineage_cells": min_cells, "min_lineages": min_lineages,
               "hvg_min_datasets": rules,
               "category_keep": keep, "category_drop": drop, "counts": counts,
               "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(outdir, "params.json"), "w"), indent=2)
    latest = os.path.join(cfg["_dirs"]["vocab"], "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(tag, latest)
    print(f"-> snapshot {outdir}  (latest -> {tag})")


def cmd_refresh(cfg, args):
    _wait(cmd_measure(cfg, args))
    cmd_build(cfg, args)


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
    return pd.DataFrame(rows, columns=["harmonized_id", "symbol", "biotype"]) \
             .drop_duplicates("harmonized_id")


def _add_flags(df, sp):
    s, bt = df["symbol"].fillna(""), df["biotype"].fillna("")
    df["is_protein_coding"] = bt == "protein_coding"
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


MGI_URL = "https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt"


def _mgi_biotype(feature):
    """MGI 'Feature Type' -> Ensembl-style biotype (protein_coding, lncRNA, pseudogene, ...)."""
    f = feature.lower()
    if "protein coding" in f:
        return "protein_coding"
    if "lncrna" in f or "lincrna" in f:
        return "lncRNA"
    if "pseudogene" in f:
        return "pseudogene"
    return re.sub(r"\s+", "_", re.sub(r"\s+gene$", "", feature.strip()))  # keeps rRNA/miRNA casing


def _parse_mgi(path):
    """Genes the rsi harmonization could only key by MGI accession (no Ensembl id)."""
    m = pd.read_csv(path, sep="\t", dtype=str, usecols=["MGI Accession ID", "Marker Symbol",
                                                         "Marker Type", "Feature Type"])
    m = m[m["Marker Type"].isin(["Gene", "Pseudogene"])]
    return pd.DataFrame({"harmonized_id": m["MGI Accession ID"], "symbol": m["Marker Symbol"],
                         "biotype": m["Feature Type"].fillna("").map(_mgi_biotype)})


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
        df = _parse_gtf(gz)
        if sp == "mouse":
            rpt = os.path.join(cfg["_dirs"]["ref"], os.path.basename(MGI_URL))
            if not os.path.exists(rpt):
                print(f"[{sp}] downloading {MGI_URL}")
                urllib.request.urlretrieve(MGI_URL, rpt)
            df = pd.concat([df, _parse_mgi(rpt)], ignore_index=True).drop_duplicates("harmonized_id")
        df = _add_flags(df, sp)
        df.to_parquet(out, index=False)
        print(f"[{sp}] genes={len(df)} protein_coding={int(df['is_protein_coding'].sum())} -> {out}")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(prog="featuresel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")

    def add_measure_opts(p):
        p.add_argument("--force", action="store_true")
        p.add_argument("--local", action="store_true", help="run here instead of via sbatch")
        p.add_argument("--jobs", type=int, help="processes for --local (default: CPUs available)")
    add_measure_opts(sub.add_parser("measure"))
    sub.add_parser("ref").add_argument("--force", action="store_true")

    def add_build_opts(p):
        p.add_argument("--hvg-min-datasets", metavar="N|KEY=N,...",
                       help="flat threshold, or per-category overrides on top of config "
                            "(e.g. protein_coding=3,lncRNA=10)")
        p.add_argument("--hvg-source", choices=HVG_SOURCES,
                       help="which per-dataset HVG call votes: union (global+lineage), "
                            "global, lineage, or upstream (the rsi pipeline's)")
        p.add_argument("--s-min", type=float, help="score (standardized variance) a gene needs in a group")
        p.add_argument("--n-max", type=int, help="rank cap within a group")
        p.add_argument("--min-lineage-cells", type=int,
                       help="ignore lineages smaller than this (>= the measurement floor)")
        p.add_argument("--min-lineages", type=int,
                       help="gene must be HVG in this many lineages for that dataset to vote")
        p.add_argument("--category-keep", help="comma-separated flags to force-include")
        p.add_argument("--category-drop", help="comma-separated flags to exclude")
        p.add_argument("--tag")
    add_build_opts(sub.add_parser("build"))
    refresh = sub.add_parser("refresh")
    add_build_opts(refresh)
    add_measure_opts(refresh)

    args = ap.parse_args()
    cfg = load_config(args.config)
    {"status": cmd_status, "measure": cmd_measure, "ref": cmd_ref,
     "build": cmd_build, "refresh": cmd_refresh}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
