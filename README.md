# eca-feature-sel

Gene-vocabulary selection for single-cell **foundation-model** training:
per-dataset HVGs, unioned across a corpus of harmonized `h5ad` files, then
filtered by gene-category keep/drop lists.

Built for HPC (Slurm): one small job per dataset, **cached**, so re-runs only
recompute what changed.

## Design

- **Two vocabularies, independent.** Human (`ENSG`) and mouse (`ENSMUSG`) are
  built in their own ID spaces — no ortholog mapping. Species comes from the
  input TSV.
- **Inputs from eca-rsi.** `scan_rsi.py` turns finished rsi units into the input
  TSV, reading species from the gene ids rather than the directory name.
- **No AnnData load, one disk read.** The worker reads `var`/`obs` and streams
  `layers/counts` with h5py; the first pass is cached in RAM (up to
  `hvg.cache_bytes`) and replayed. Dense files are converted to stored entries on
  read (they are ~97% zeros), so a 1.5 GB dense matrix costs 0.5 s.
- **Numerical guards, not QC.** Cells under `min_cell_counts` total counts and,
  per group, genes seen in fewer than `min_gene_cells` cells are excluded: those
  identical (mean, var) points are what made loess singular.
- **A dataset's HVG = global ∪ per-lineage.** `seurat_v3` (vst) is scored on the
  whole dataset and again inside each coarse lineage (`zmip_ann_coarse`), because
  the global call is dominated by the majority population. Implemented here
  (`hvg.py`), not via scanpy: streaming passes, loess with a quadratic fallback.
  Verified against scanpy (Jaccard > 0.95; Pearson residuals to 1e-13).
- **Measure once, threshold later.** Measurement stores every gene's *score*
  (standardized variance; 1 = as variable as a typical gene of that expression
  level) and *rank* per group. A gene passes in a group when
  `score >= s_min` and `rank <= n_max`; both are build-time knobs. No fixed
  "top 2000": at rank 2000 the score is already ~1.1 on a 28k-cell dataset.
- **One rule, per-category thresholds.** A gene is a candidate iff it is HVG in
  enough datasets, where "enough" depends on its category (e.g. `protein_coding: 3`,
  `lncRNA: 10`). `N` emerges from the thresholds; it is not preset.
- **Categories after data.** `category_keep` force-includes (whitelist),
  `category_drop` excludes (blacklist); keep wins over drop.
- **Stats are descriptive.** Detection rate and mean counts are reported per
  gene so you can see how strongly a selected gene is expressed — they do not
  filter anything.

See `docs/selection_design.md` for the rules, flags and stat columns.

## Usage

```bash
PY=/path/to/venv/bin/python          # needs anndata scanpy scikit-misc numpy pandas pyarrow scipy yaml
# edit config.yaml: inputs_tsv, cache_root (MUST be on $SCRATCH), venv_python

$PY scan_rsi.py                      # rsi results -> mouse.tsv (--species human, --list)
$PY featuresel.py status             # what's done / stale / missing
$PY featuresel.py ref                # one-time: Ensembl biotype reference (needs internet)
$PY featuresel.py measure --local    # all CPUs of this node; 160 mouse datasets in ~3 min
$PY featuresel.py measure            # or one Slurm array task per dataset
$PY featuresel.py build              # union HVGs + category rules -> snapshot
$PY featuresel.py refresh            # measure -> wait -> build

# retune per category (build is instant; measurement is untouched):
$PY featuresel.py build --hvg-min-datasets lncRNA=15,miRNA=5 --tag t2
$PY featuresel.py build --hvg-min-datasets 3 --tag flat3          # bare int = flat
$PY featuresel.py build --category-drop is_pseudogene,is_OR,is_vomeronasal,is_taste --tag narrow
$PY featuresel.py build --category-keep is_protein_coding --tag allpc  # unconditional
$PY featuresel.py build --s-min 1.2 --n-max 2000 --tag loose   # per-group pass rule
$PY featuresel.py build --hvg-source global --tag noline       # drop the lineage half
$PY featuresel.py build --min-lineage-cells 500 --min-lineages 2 --tag strictlin

$PY report.py --all                  # per-dataset HTML report + genes.tsv + index.html
$PY explore.py mca3.0-brain-mouse    # interactive explorer (sliders, vst vs pearson)
$PY votes.py                         # corpus vote explorer: genes passing per category,
                                     # s_min and vote cutoff as sliders -> cache/figs/votes_{sp}.html
```

Each `build` writes `cache/vocab/<tag>/`: `vocab_{sp}.tsv` (key column =
`harmonized_id`, the FM token id), two tuning tables — `thresholds_{sp}.tsv`
(genes per biotype at each dataset-count threshold) and `sweep_smin_{sp}.tsv`
(the whole selection re-run at s_min 1.1…2.0: votes per dataset and final size)
— and `params.json`; and updates `cache/vocab/latest`.

## Layout

```
featuresel.py   CLI (status / measure / ref / build / refresh)
worker.py       per-dataset scores + expression stats (one process per dataset)
hvg.py          streaming vst / pearson scoring, dataset-level + per lineage
report.py       per-dataset HTML report from the cached measurement
explore.py      interactive single-dataset explorer (plotly, self-contained)
scan_rsi.py     eca-rsi results -> inputs TSV
test_worker.py  self-check: python test_worker.py
config.yaml     paths + slurm resources + hvg/selection defaults
human.tsv       human inputs: sample_key, species, h5ad (tab-separated)
mouse.tsv       mouse inputs
# cache_root (on $SCRATCH, git-ignored): stats/  ref/  vocab/<tag>/  jobs/
```

## Notes
- Each input TSV has three tab-separated columns: `sample_key`, `species`,
  `h5ad`. `#` comments and a header row are allowed; relative h5ad paths resolve
  against the TSV.
- Each dataset writes two parquets: per-gene stats with the dataset-level
  `score`/`rank`, and a `.lineages.parquet` with (lineage, lineage_cells, gene,
  score, rank) for the top `hvg.n_store` genes of each lineage. So `--s-min`,
  `--n-max`, `--min-lineage-cells`, `--min-lineages` are all **build-time** —
  retune without remeasuring. `hvg.min_lineage_cells` is only the measurement
  floor and `n_store` the ceiling on `n_max`.
- `hvg.method: pearson` (analytic Pearson residuals) is implemented and verified
  but O(cells × genes) dense — ~50× slower than vst. Kept for comparison in
  `explore.py`.
- The file's own `var.highly_variable` is carried along as `hvg_upstream`
  (`--hvg-source upstream`) for comparison. Set `hvg.compute: false` to read only
  that and skip our passes.
- Each h5ad needs `layers/counts` and `var.gene_id_harmonized`. Var rows with no
  harmonized id are dropped; duplicate ids are summed into one column before
  stats and HVG.
- Reuse is mtime-based plus a hash of the `hvg:` config block: a dataset is
  recomputed only if its h5ad is newer or the HVG settings changed. Dropping a
  dataset from the TSV drops it from the next build.
- Jobs are small and I/O bound (the streaming pass never holds the matrix);
  `slurm.mem` only needs headroom for `hvg.compute: true`.
