# eca-feature-sel

Counts-driven gene-vocabulary selection for single-cell **foundation-model**
training. Turns a large corpus of harmonized `h5ad` datasets into a fixed,
cross-dataset gene vocabulary — **from the data** — and lets you decide the gene
range interactively over time.

Built for HPC (Slurm): measurement fans out one small job per dataset and is
**cached**, so periodic re-runs only recompute what changed.

## Design

- **Two vocabularies, independent.** Human (Ensembl `ENSG`) and mouse (`ENSMUSG`)
  are built separately in their own ID spaces — no ortholog mapping. Species is
  read from each dataset's `curation_stats.json`.
- **Data-first.** Detection from raw `layers/counts` (the CSR/CSC nonzero pattern)
  is the primary axis; `.var` biotype is auxiliary. The globally-near-zero tail is
  cut by the data.
- **Measurement ≠ policy.** Stage 1 computes and caches per-gene stats and *deletes
  nothing*. Selection (Stage 3) is instant and re-runnable: it applies thresholds
  and — only if you ask — vetoes specific gene families.

### Inclusion rule
A harmonized gene is a **candidate** if either arm holds:
- **consistency** — detected (`det ≥ f`) in `≥ k` datasets (default f=1%, k=3);
- **strength** — `max_det ≥ 0.25` and `max_mean_expr ≥` q0.90 (rescues real
  tissue-restricted markers like `INS`/`SFTPC` the consistency arm would miss).

`N` emerges from the thresholds; it is not preset.

### Biotype veto (off by default)
Every gene is annotated (biotype + family flags: OR, IG/TR V·D·J·C, taste,
vomeronasal, mt, hb, ribo, sex) and each snapshot reports what each scenario costs:
- **narrow** — drop OR / vomeronasal / taste / IG-V·D·J / TR-V·D·J; **keep** IG-C/TR-C
  constant regions. Robust lncRNA (MALAT1/NEAT1/XIST) survive on data alone.
- **wide** — narrow + drop all non-`protein_coding` (also drops MALAT1/NEAT1/XIST).

Turn a scenario on with `--veto`; the `veto_narrow`/`veto_wide` columns are always
present so you can also just filter the TSV.

## Usage

```bash
PY=/path/to/venv/bin/python          # needs h5py numpy pandas pyarrow yaml
# edit config.yaml: corpus_root, cache_root (MUST be on $SCRATCH), venv_python

$PY featuresel.py status             # what's done / stale / missing
$PY featuresel.py ref                # one-time: Ensembl biotype reference (needs internet)
$PY featuresel.py measure            # Slurm job array over stale/new datasets (reuses rest)
$PY featuresel.py build              # aggregate cached stats + select -> versioned snapshot
$PY featuresel.py refresh            # measure -> wait -> build, one shot

# decide the gene range:
$PY featuresel.py build --k 2 --tag loose         # try a looser threshold
$PY featuresel.py build --veto is_OR,is_IG_V,is_IG_D,is_IG_J,is_TR_V,is_TR_D,is_TR_J,is_vomeronasal,is_taste --tag narrow
$PY featuresel.py list                            # all snapshots
$PY featuresel.py diff latest narrow              # genes added / removed between snapshots
```

Each `build` writes `cache/vocab/<tag>/vocab_{human,mouse}.tsv` (key column =
`harmonized_id`, the FM token id) + `params.json`, and updates `cache/vocab/latest`.

## Layout

```
featuresel.py   CLI (status/measure/ref/build/refresh/diff/list)
worker.py       per-dataset Stage-1 (streaming h5py, CSR+CSC), run by each array task
config.yaml     paths + slurm resources + default policy
# cache_root (on $SCRATCH, git-ignored): stage1/  ref/  master/  vocab/<tag>/  jobs/
```

## Notes
- Reuse is mtime-based: a dataset is recomputed only if its `h5ad` is newer than its
  cached stat (or missing). Removing a dataset from the corpus drops it from the next
  build automatically.
- Per-dataset jobs are tiny (streaming bincount, <1 GB peak) — they queue fast and
  spread across nodes.
- Not a coverage fix: mouse lncRNA are limited upstream (Tabula Muris quantified a
  ~23k-gene, protein-coding-focused reference), not by cell count.
