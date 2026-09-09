#!/usr/bin/env bash
# End to end: rsi results -> per-dataset scores -> corpus vote -> final gene list.
# Everything is cached; rerunning only measures new or changed h5ad files.
#
#   ./pipeline.sh [tag]        default tag: final
#
# Rules live in config.yaml (selection:). Current final rule:
#   score >= s_min 1.3 in the dataset or in one of its lineages = one vote;
#   protein_coding needs >= 3 datasets, every other biotype >= 10;
#   then category_keep / category_drop.
set -euo pipefail
cd "$(dirname "$0")"
PY=$(sed -n 's/^venv_python: *\([^ #]*\).*/\1/p' config.yaml)
TAG=${1:-final}

"$PY" scan_rsi.py                       # finished rsi units -> mouse.tsv
"$PY" featuresel.py ref                 # biotype reference (no-op once built)
"$PY" featuresel.py measure --local     # per-dataset scores, all CPUs of this node
"$PY" featuresel.py build --tag "$TAG"  # vote + category rules -> snapshot
"$PY" votes.py                          # corpus vote explorer (cache/figs/votes_*.html)

ROOT=$(sed -n 's/^cache_root: *\([^ #]*\).*/\1/p' config.yaml)
echo
echo "final gene list(s) (genes_*.tsv = selected only; vocab_*.tsv = every gene with stats):"
ls "$ROOT/vocab/$TAG"/genes_*.tsv
