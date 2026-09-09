#!/usr/bin/env python
"""Build an inputs TSV from eca-rsi results.

Finds every completed unit (``<dataset>/.../rsi/units/<unit>/release/final.h5ad``)
under a corpus root, reads the species from the harmonized gene ids in the file
itself (ENSMUSG / ENSG -- never from the directory name), and writes the
sample_key/species/h5ad TSV that featuresel.py consumes. rsi runs that have no
`release/final.h5ad` yet (still running, or never started) are listed, not
written, so re-running this after they finish picks them up.

    scan_rsi.py                      # mouse -> mouse.tsv
    scan_rsi.py --species human --out human_rsi.tsv
    scan_rsi.py --list               # report only, write nothing
"""
import argparse
import glob
import os
import re
from collections import Counter

import h5py

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


def find_units(root):
    pats = [f"{root}/{'*/' * d}rsi/units/*/release/final.h5ad" for d in (1, 2, 3)]
    return sorted({p for pat in pats for p in glob.glob(pat)})


def find_incomplete(root):
    pats = [f"{root}/{'*/' * d}rsi" for d in (1, 2, 3)]
    out = []
    for rsi in sorted({p for pat in pats for p in glob.glob(pat)}):
        units = glob.glob(f"{rsi}/units/*/")
        done = glob.glob(f"{rsi}/units/*/release/final.h5ad")
        if len(done) < len(units) or not units:
            status = os.path.join(rsi, "status.txt")
            last = open(status).read().strip().splitlines()[-1] if os.path.exists(status) else ""
            out.append((rsi, len(units), len(done), last))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--species", default="mouse", choices=sorted(set(PREFIX.values())))
    ap.add_argument("--out", help="default: <species>.tsv next to this script")
    ap.add_argument("--list", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    rows, bad = [], []
    for path in find_units(a.root):
        info, err = peek(path)
        if err:
            bad.append((path, err))
            continue
        sp, shape = info
        rows.append({"key": sample_key(a.root, path), "species": sp,
                     "path": path, "shape": shape})

    dup = [k for k, n in Counter(r["key"] for r in rows).items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate sample_key(s): {dup}")

    print(f"{len(rows)} completed unit(s) under {a.root}: "
          f"{dict(Counter(r['species'] for r in rows))}")
    for path, err in bad:
        print(f"  skipped ({err}): {path}")
    incomplete = find_incomplete(a.root)
    if incomplete:
        print(f"  {len(incomplete)} rsi run(s) with no finished unit yet:")
        for rsi, n_units, n_done, last in incomplete:
            print(f"    units={n_units} final={n_done}  {rsi}  {last}")

    keep = sorted([r for r in rows if r["species"] == a.species], key=lambda r: r["key"])
    cells = sum(r["shape"][0] for r in keep)
    print(f"\n[{a.species}] {len(keep)} dataset(s), {cells:,} cells, "
          f"{min((r['shape'][1] for r in keep), default=0)}-"
          f"{max((r['shape'][1] for r in keep), default=0)} genes")
    print("  by source: " + ", ".join(f"{k}={v}" for k, v in
                                      Counter(r["key"].split("-")[0] for r in keep).most_common()))
    if a.list:
        return
    out = a.out or os.path.join(HERE, f"{a.species}.tsv")
    with open(out, "w") as fh:
        fh.write("sample_key\tspecies\th5ad\n")
        for r in keep:
            fh.write(f"{r['key']}\t{r['species']}\t{r['path']}\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
