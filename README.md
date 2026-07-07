# eca-feature-sel

Data-driven gene-vocabulary selection for single-cell **foundation-model**
training. Turns a large corpus of harmonized `h5ad` datasets into a fixed,
cross-dataset gene vocabulary using detection, HVG, and cluster-marker evidence.

Built for HPC (Slurm): measurement fans out one small job per dataset and is
**cached**, so periodic re-runs only recompute what changed.

## Design

- **Two vocabularies, independent.** Human (Ensembl `ENSG`) and mouse (`ENSMUSG`)
  are built separately in their own ID spaces — no ortholog mapping. Species is
  read from each dataset's `curation_stats.json`.
- **Data-first.** Detection from raw `layers/counts`, per-dataset HVGs, and
  within-dataset cluster markers are used as inclusion evidence. `.var` biotype is
  auxiliary and is applied only by the category rule.
- **Measurement ≠ policy.** Stage 1 caches per-gene detection stats. Stage 2 caches
  per-dataset HVG and cluster-DEG selections. Selection is instant and re-runnable:
  it unions the cached evidence and applies a category rule.

### Inclusion rule
A harmonized gene is a **candidate** if any v2 inclusion rule holds:
- **detection** — detected in at least `min_frac` cells in at least
  `min_dataset_occurrence` datasets (default `min_frac=0.01`,
  `min_dataset_occurrence=10`);
- **HVG** — selected as a highly variable gene within at least `min_datasets`
  datasets (default top 3,000 HVGs per dataset);
- **cluster-DEG** — selected as a one-vs-rest marker for at least one eligible
  cell-type or cluster group.

`N` emerges from the thresholds; it is not preset.

### Category rule
Every gene is annotated (biotype + family flags: OR, IG/TR V·D·J·C, taste,
vomeronasal, mt, hb, ribo, sex) and each snapshot reports what each scenario costs:
- **default** — drop pseudogenes after count-driven candidate selection.
- **narrow** — drop OR / vomeronasal / taste / IG-V·D·J / TR-V·D·J; **keep** IG-C/TR-C
  constant regions. Robust lncRNA (MALAT1/NEAT1/XIST) survive on data alone.
- **wide** — narrow + drop all non-`protein_coding` (also drops MALAT1/NEAT1/XIST).

Set categories with `--category-rule`; the legacy `--veto` alias still works. The
`category_excluded`, `selected_by_detection`, `selected_by_hvg`,
`selected_by_cluster_deg`, `veto_narrow`, and `veto_wide` columns are written so
you can also filter the TSV directly.

## Usage

```bash
PY=/path/to/venv/bin/python          # needs h5py numpy pandas pyarrow yaml scanpy scipy
# edit config.yaml: inputs_tsv, cache_root (MUST be on $SCRATCH), venv_python

$PY featuresel.py status             # what's done / stale / missing
$PY featuresel.py ref                # one-time: Ensembl biotype reference (needs internet)
$PY featuresel.py measure            # Slurm job array over stale/new datasets (reuses rest)
$PY featuresel.py measure-v2         # Slurm job array for HVG + cluster-DEG cache
$PY featuresel.py measure-v2-local --limit 4    # local smoke/profiling run, one Python process
$PY featuresel.py build-v2           # union detection/HVG/cluster-DEG -> snapshot
$PY featuresel.py refresh-v2         # measure + measure-v2 -> wait -> build-v2

# decide the gene range:
$PY featuresel.py build-v2 --min-dataset-occurrence 8 --tag loose
$PY featuresel.py build-v2 --category-rule is_pseudogene,is_OR,is_IG_V,is_IG_D,is_IG_J,is_TR_V,is_TR_D,is_TR_J,is_vomeronasal,is_taste --tag narrow
$PY featuresel.py list                            # all snapshots
$PY featuresel.py diff latest narrow              # genes added / removed between snapshots
```

Each `build-v2` writes `cache/vocab/<tag>/vocab_{human,mouse}.tsv` (key column =
`harmonized_id`, the FM token id) + `params.json`, and updates `cache/vocab/latest`.
The legacy `build` command is still available for older detection/strength
snapshots.

## Layout

```
featuresel.py   CLI (status/measure/measure-v2/ref/build-v2/refresh-v2/diff/list)
worker.py       per-dataset Stage-1 detection stats, run by each array task
worker_v2.py    per-dataset Stage-2 HVG + cluster-DEG selections
config.yaml     paths + slurm resources + default policy
human.tsv       explicit human input dataset list: sample_key, species, h5ad
mouse.tsv       explicit mouse input dataset list: sample_key, species, h5ad
# cache_root (on $SCRATCH, git-ignored): stage1/  stage2/  ref/  master/  vocab/<tag>/  jobs/
```

## Notes
- Inputs are explicit when `inputs_tsv` is set. It can be one TSV path or a YAML list
  of TSV paths, for example `human.tsv` and `mouse.tsv`. Each TSV must have three
  tab-separated columns: `sample_key`, `species`, and `h5ad`. Lines beginning with `#`
  are ignored; a header row is allowed. Relative h5ad paths are resolved relative to
  the TSV file.
- If `inputs_tsv` is unset, corpus discovery falls back to scanning immediate dataset
  directories under `corpus_root` for usable `.h5ad` files. This fallback does not
  assume an h5ad filename suffix; files are chosen by content.
- Reuse is mtime-based: a dataset is recomputed only if its selected `h5ad` is newer
  than its cached stat (or missing). Removing a dataset from the corpus drops it from
  the next build automatically.
- Stage-1 jobs are tiny (streaming bincount, <1 GB peak). Stage-2 jobs load one
  dataset at a time with Scanpy and use the separate `slurm_v2` resources.
- Stage-2 meta JSON files include `phase_times_sec` so import, read, HVG, and
  cluster-DEG bottlenecks can be profiled after a run. `measure-v2-local` processes
  multiple datasets in one Python process to avoid repeated Scanpy import overhead
  during local smoke tests.
- Not a coverage fix: mouse lncRNA are limited upstream (Tabula Muris quantified a
  ~23k-gene, protein-coding-focused reference), not by cell count.
