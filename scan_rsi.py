#!/usr/bin/env python
"""Build an inputs TSV from eca-rsi results.

Finds every completed unit (``<dataset>/.../rsi/units/<unit>/release/final.h5ad``)
under a corpus root, reads the species from the harmonized gene ids in the file
itself (ENSMUSG / ENSG -- never from the directory name), and writes the
sample_key/species/h5ad TSV that featuresel.py consumes. rsi runs that have no
`release/final.h5ad` yet (still running, or never started) are listed, not
written, so re-running this after they finish picks them up.

Two layouts are understood, under any number of roots (``--root``, repeatable):

    <root>/<dataset>/.../<group>/rsi/units/<unit>/release/final.h5ad   (Oak)
    <root>/<batch>/<run>/units/<unit>/release/final.h5ad               (gen2 runs on scratch)

Keys listed in ``exclude.txt`` (one sample_key per line, ``#`` comments) are
left out, e.g. an Oak run that was re-processed on scratch.

Every filesystem call runs in a forked child with a deadline (Oak can hang a
single open() for minutes). Whatever does not answer in time is *deferred*: its
row from the previous TSV is kept as is, and the next run checks it again.

    scan_rsi.py                      # mouse -> mouse.tsv
    scan_rsi.py --species human --out human_rsi.tsv
    scan_rsi.py --list               # report only, write nothing
    scan_rsi.py --timeout 300        # be more patient with Oak
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

import h5py

import featuresel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOTS = [os.path.expanduser("~/oak/data/sc"),
                 os.path.join(os.environ.get("SCRATCH", "/scratch/users/chensj16"),
                              "eca-runs/gen2-acceptance-20260917")]
PREFIX = {"ENSMUSG": "mouse", "ENSG": "human"}


def norm(name):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def sample_key(root, run, multi=False):
    """Oak layout <dataset>/.../<group>/rsi: <dataset>-<group>-<unit> (group dropped
    when it repeats a neighbour). Batch layout <batch>/<run> with a spec.json:
    the run's dataset_id, e.g. 'Tabula Sapiens / Ear' -> tabula-sapiens-ear,
    plus the unit when the run has several."""
    keys = []
    for path in run["units"]:
        parts = os.path.relpath(path, root).split("/")
        u = parts.index("units")
        unit = parts[u + 1]
        if parts[u - 1] == "rsi":
            top, group = parts[0], norm(parts[u - 2])
            keep_group = not (group == top or group in unit or unit in group)
            base = [top] + ([group] if keep_group else []) + [unit]
        else:
            base = [norm(run["dataset_id"] or parts[u - 1])] + ([unit] if multi else [])
        keys.append(re.sub(r"-+", "-", "-".join(base)))
    return keys


def rel_dataset(path):
    """Path of a dataset dir relative to the corpus root, whichever alias of Oak is used."""
    return path.split("/data/sc/", 1)[1] if "/data/sc/" in path else path
def peek(path):
    """(species, n_cells, n_genes) from metadata only; species from the gene ids."""
    with h5py.File(path, "r") as f:
        if "layers/counts" not in f:
            return None, "no layers/counts"
        counts = f["layers/counts"]
        shape = counts.attrs.get("shape")  # sparse group; dense arrays carry .shape
        shape = tuple(int(x) for x in (shape if shape is not None else counts.shape))
        node = f.get("var/gene_id_harmonized")
        if node is None:
            return None, "no var/gene_id_harmonized"
        vals = node["categories"][:50] if isinstance(node, h5py.Group) else node[:50]
        vals = [v.decode() if isinstance(v, bytes) else str(v) for v in vals]
        for v in vals:
            for pre, sp in PREFIX.items():
                if v.startswith(pre):
                    return (sp, shape), None
        return None, f"unrecognized gene ids ({vals[:2]})"


def scan_top(top):
    """One top-level dir -> list of runs. A run is any directory holding units/
    (an Oak .../<group>/rsi or a scratch <batch>/<run>); units/ dirs nested inside
    another run (e.g. <run>/00-organize/units) are stages, not runs."""
    cands = sorted({os.path.dirname(p) for d in (0, 1, 2, 3) for p in glob.glob(f"{top}/{'*/' * d}units")})
    runs = [r for r in cands if not any(r.startswith(o + "/") for o in cands if o != r)]
    out = []
    for run in runs:
        all_units = glob.glob(f"{run}/units/*/")
        done = sorted(glob.glob(f"{run}/units/*/release/final.h5ad"))
        rec = {"run": run, "units": done, "n_units": len(all_units), "dataset_id": None,
               "input_root": None, "mtime": max((os.path.getmtime(p) for p in done), default=0)}
        spec = os.path.join(run, "spec.json")
        if os.path.exists(spec):
            try:
                d = json.load(open(spec))
                rec["dataset_id"], rec["input_root"] = d.get("dataset_id"), d.get("input_root")
            except ValueError:
                pass
        if len(done) < len(all_units) or not all_units:
            status = os.path.join(run, "status.txt")
            rec["last"] = open(status).read().strip().splitlines()[-1] if os.path.exists(status) else ""
        out.append(rec)
    return out


def read_tsv(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {r[0]: r for r in csv.reader(fh, delimiter="\t") if r and r[0] != "sample_key"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help=f"repeatable; default {DEFAULT_ROOTS}")
    ap.add_argument("--exclude", default=os.path.join(HERE, "exclude.txt"),
                    help="sample_keys to leave out, one per line, # comments")
    ap.add_argument("--species", default="mouse", choices=sorted(set(PREFIX.values())))
    ap.add_argument("--out", help="default: <species>.tsv next to this script")
    ap.add_argument("--list", action="store_true", help="report only, write nothing")
    ap.add_argument("--timeout", type=int, default=featuresel.FS_TIMEOUT,
                    help="seconds each stage of filesystem calls gets before the rest is deferred")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, f"{a.species}.tsv")
    old = read_tsv(out)

    roots = a.root or DEFAULT_ROOTS
    excluded = {}
    if os.path.exists(a.exclude):
        for line in open(a.exclude):
            key, _, why = line.partition("#")
            if key.strip():
                excluded[key.strip()] = why.strip()

    # stage 0: each root listing; stage 1: each top-level dir; stage 2: each h5ad
    res, dfr = featuresel.fs_parallel(os.listdir, roots, a.timeout)
    if dfr:
        raise SystemExit(f"cannot even list {dfr} within {a.timeout}s; try later")
    tops = sorted(os.path.join(r, d) for r in roots for d in res[r] if not d.startswith("."))
    res, slow_tops = featuresel.fs_parallel(scan_top, tops, a.timeout)
    runs = [r for recs in res.values() for r in recs]
    incomplete = sorted((r["run"], r["n_units"], len(r["units"]), r["last"]) for r in runs if "last" in r)
    units = sorted(p for r in runs for p in r["units"])
    res, slow_files = featuresel.fs_parallel(peek, units, a.timeout)

    # a gen2 run re-processed an Oak dataset: its input_root sits in that dataset's
    # dir, so the Oak run there is superseded (the gen2 one is the newer pipeline)
    reprocessed = {rel_dataset(os.path.dirname(r["input_root"])): r for r in runs
                   if r["input_root"] and r["units"]}
    rows, bad, left_out, superseded = [], [], [], []
    for run in runs:
        root = next(rt for rt in roots if run["run"].startswith(rt + "/"))
        keys = sample_key(root, run, multi=len(run["units"]) > 1)
        for path, key in zip(run["units"], keys):
            if path in slow_files:
                continue
            info, err = res.get(path, (None, "unreadable"))
            if err:
                bad.append((path, err))
                continue
            sp, shape = info
            if os.path.basename(run["run"]) == "rsi":
                newer = reprocessed.get(rel_dataset(os.path.dirname(run["run"])))
                if newer:
                    superseded.append((key, sp, newer["dataset_id"]))
                    continue
            if key in excluded:
                left_out.append((key, sp, excluded[key]))
                continue
            rows.append({"key": key, "species": sp, "path": path, "shape": shape, "mtime": run["mtime"]})

    # the same dataset finished twice in one batch (a retried run): keep the newest
    by_key = {}
    for r in rows:
        if r["key"] not in by_key or r["mtime"] > by_key[r["key"]]["mtime"]:
            by_key[r["key"]] = r
    retried = [r for r in rows if by_key[r["key"]] is not r]
    rows = list(by_key.values())

    dup = [k for k, n in Counter(r["key"] for r in rows).items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate sample_key(s): {dup}")

    print(f"{len(rows)} completed unit(s) under {roots}: "
          f"{dict(Counter(r['species'] for r in rows))}")
    for path, err in bad:
        print(f"  skipped ({err}): {path}")
    for key, sp, by in superseded:
        print(f"  superseded ({sp}) {key}: re-processed by gen2 run '{by}'")
    for r in retried:
        print(f"  older duplicate run dropped ({r['species']}) {r['key']}: {r['path']}")
    for key, sp, why in left_out:
        print(f"  excluded ({sp}) {key}: {why}")
    if incomplete:
        print(f"  {len(incomplete)} rsi run(s) with no finished unit yet:")
        for run, n_units, n_done, last in incomplete:
            print(f"    units={n_units} final={n_done}  {run}  {last}")

    # deferred: keep whatever the previous TSV said about them
    kept = {}
    for key, row in old.items():
        p = row[2]
        if p in slow_files or any(p.startswith(t + "/") for t in slow_tops):
            kept[key] = row
    if slow_tops or slow_files:
        print(f"  DEFERRED (no answer within {a.timeout}s): {len(slow_tops)} dataset dir(s) "
              f"{[os.path.basename(t) for t in slow_tops]}, {len(slow_files)} file(s) "
              f"{[os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p))))) for p in slow_files]}; "
              f"kept {len(kept)} row(s) from the previous {os.path.basename(out)}. Re-run later.")

    keep = sorted([r for r in rows if r["species"] == a.species], key=lambda r: r["key"])
    cells = sum(r["shape"][0] for r in keep)
    print(f"\n[{a.species}] {len(keep)} dataset(s) seen now, {cells:,} cells, "
          f"{min((r['shape'][1] for r in keep), default=0)}-"
          f"{max((r['shape'][1] for r in keep), default=0)} genes"
          + (f" (+{len(kept)} kept from before)" if kept else ""))
    print("  by source: " + ", ".join(f"{k}={v}" for k, v in
                                      Counter(r["key"].split("-")[0] for r in keep).most_common()))
    if not a.list:
        lines = {r["key"]: f"{r['key']}\t{r['species']}\t{r['path']}" for r in keep}
        lines.update({k: "\t".join(v) for k, v in kept.items() if k not in lines and v[1] == a.species})
        with open(out, "w") as fh:
            fh.write("sample_key\tspecies\th5ad\n")
            for k in sorted(lines):
                fh.write(lines[k] + "\n")
        print(f"-> {out}")
    featuresel.fs_exit(0)


if __name__ == "__main__":
    main()
