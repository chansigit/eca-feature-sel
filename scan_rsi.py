#!/usr/bin/env python
"""Build an inputs TSV from eca-rsi results.

Finds every completed unit (``<dataset>/.../rsi/units/<unit>/release/final.h5ad``)
under a corpus root, reads the species from the harmonized gene ids in the file
itself (ENSMUSG / ENSG -- never from the directory name), and writes the
sample_key/species/h5ad TSV that featuresel.py consumes. rsi runs that have no
`release/final.h5ad` yet (still running, or never started) are listed, not
written, so re-running this after they finish picks them up.

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
import os
import re
from collections import Counter

import h5py

import featuresel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.expanduser("~/oak/data/sc")
PREFIX = {"ENSMUSG": "mouse", "ENSG": "human"}


def sample_key(root, path):
    """<dataset>-<group>-<unit>, dropping the group when it repeats a neighbour."""
    parts = os.path.relpath(path, root).split("/")
    top = parts[0]
    group = parts[parts.index("rsi") - 1].lower().replace("_", "-")
    unit = parts[parts.index("units") + 1]
    keep_group = not (group == top or group in unit or unit in group)
    return re.sub(r"-+", "-", "-".join([top] + ([group] if keep_group else []) + [unit]))


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
    """One top-level dataset dir -> (finished unit paths, unfinished rsi runs)."""
    rsis = sorted({p for d in (0, 1, 2) for p in glob.glob(f"{top}/{'*/' * d}rsi")})
    units, incomplete = [], []
    for rsi in rsis:
        all_units = glob.glob(f"{rsi}/units/*/")
        done = sorted(glob.glob(f"{rsi}/units/*/release/final.h5ad"))
        units += done
        if len(done) < len(all_units) or not all_units:
            status = os.path.join(rsi, "status.txt")
            last = open(status).read().strip().splitlines()[-1] if os.path.exists(status) else ""
            incomplete.append((rsi, len(all_units), len(done), last))
    return units, incomplete


def read_tsv(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {r[0]: r for r in csv.reader(fh, delimiter="\t") if r and r[0] != "sample_key"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--species", default="mouse", choices=sorted(set(PREFIX.values())))
    ap.add_argument("--out", help="default: <species>.tsv next to this script")
    ap.add_argument("--list", action="store_true", help="report only, write nothing")
    ap.add_argument("--timeout", type=int, default=featuresel.FS_TIMEOUT,
                    help="seconds each stage of filesystem calls gets before the rest is deferred")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, f"{a.species}.tsv")
    old = read_tsv(out)

    # stage 0: the root listing; stage 1: each dataset dir; stage 2: each h5ad
    res, dfr = featuresel.fs_parallel(os.listdir, [a.root], a.timeout)
    if dfr:
        raise SystemExit(f"cannot even list {a.root} within {a.timeout}s; try later")
    tops = sorted(os.path.join(a.root, d) for d in res[a.root] if not d.startswith("."))
    res, slow_tops = featuresel.fs_parallel(scan_top, tops, a.timeout)
    units = sorted(p for u, _ in res.values() for p in u)
    incomplete = sorted(i for _, inc in res.values() for i in inc)
    res, slow_files = featuresel.fs_parallel(peek, units, a.timeout)

    rows, bad = [], []
    for path in units:
        if path in slow_files:
            continue
        info, err = res.get(path, (None, "unreadable"))
        if err:
            bad.append((path, err))
            continue
        sp, shape = info
        rows.append({"key": sample_key(a.root, path), "species": sp, "path": path, "shape": shape})

    dup = [k for k, n in Counter(r["key"] for r in rows).items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate sample_key(s): {dup}")

    print(f"{len(rows)} completed unit(s) under {a.root}: "
          f"{dict(Counter(r['species'] for r in rows))}")
    for path, err in bad:
        print(f"  skipped ({err}): {path}")
    if incomplete:
        print(f"  {len(incomplete)} rsi run(s) with no finished unit yet:")
        for rsi, n_units, n_done, last in incomplete:
            print(f"    units={n_units} final={n_done}  {rsi}  {last}")

    # deferred: keep whatever the previous TSV said about them
    kept = {}
    for key, row in old.items():
        p = row[2]
        if p in slow_files or any(p.startswith(t + "/") for t in slow_tops):
            kept[key] = row
    if slow_tops or slow_files:
        print(f"  DEFERRED (no answer within {a.timeout}s): {len(slow_tops)} dataset dir(s) "
              f"{[os.path.basename(t) for t in slow_tops]}, {len(slow_files)} file(s) "
              f"{[sample_key(a.root, p) for p in slow_files]}; "
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
