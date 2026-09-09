# Selection Design

## Goal

A gene enters the vocabulary if it is highly variable in enough individual
datasets. Gene-category keep/drop lists are applied afterwards. Everything else
(cross-dataset detection thresholds, cluster markers) was removed: it added
policy knobs without changing the vocabulary much, and cost most of the compute.

## Rules

1. **HVG union, per-category threshold** — each dataset contributes its HVGs.
   Two HVG calls are measured per dataset and `hvg_source` picks which one the
   union uses:

   - `union` (default) — the dataset-level `seurat_v3` call **union** the call
     run separately inside each coarse lineage (`hvg.lineage_obs_key`, lineages
     under `min_lineage_cells` skipped). A dataset votes for a gene if either
     found it.
   - `global` / `lineage` — either half alone.
   - `upstream` — `var.highly_variable` as the rsi pipeline left it (`seurat`
     flavor, 2000 genes, batch-aware). Free to read; kept for comparison.

   **Why the union.** The two halves capture orthogonal variance: the global call
   ranks genes by total variance, which is dominated by between-lineage identity
   differences; a lineage-internal call finds state/zonation genes that a
   majority population would otherwise mask. Measured on mca3.0 Brain (28472
   cells, 5 lineages >= 500 cells):

   | | genes |
   |---|---|
   | global top-2000 | 2000 |
   | union of 5 lineage calls | 6564 |
   | overlap | 1517 |
   | in *no* lineage call | 483 |

   Neither side is redundant. And widening the global call does not substitute:
   global top-6564 covers only 50% of the lineage union; you need top-19692 (half
   the transcriptome) to reach 91%.

   Per-lineage calls on a few hundred cells are noisy on their own -- the noise
   filter is the cross-dataset threshold below, not the per-dataset call. A gene
   picked by chance in one lineage will not be picked in three datasets.

   **Within a group, what counts as HVG.** Not a fixed top-N. Each gene gets a
   *score* = standardized variance (Seurat v3 `vst.variance.standardized`,
   scanpy `variances_norm`): its variance divided by the loess-expected variance
   of a typical gene at that mean, after clipping each cell at √N. 1 means
   "typical", 2 means twice the variance. A gene passes in a group when
   `score >= s_min` (default 1.5) and `rank <= n_max` (default 3000, a cap that
   only binds on small or plate-based datasets). The score is comparable across
   groups and datasets, ranks are not; at rank 2000 the score is already ~1.1 on
   a 28k-cell dataset, i.e. a fixed 2000 mostly selects trend-line genes.

   A dataset votes for a gene if it passes in the dataset-level group OR in
   `min_lineages` lineages of at least `min_lineage_cells` cells; one vote per
   dataset however many groups hit.

   A gene enters the union when its vote count reaches the threshold *for its
   category*:

   ```yaml
   hvg_min_datasets:
     protein_coding: 3      # first match wins, so narrow keys go on top
     lncRNA: 10
     default: 10
   ```

   Keys are `is_*` flags or exact Ensembl biotypes; `default` covers the rest.
   The resolved threshold is written per gene as `min_datasets_required`.
2. **Category keep (whitelist)** — a gene matching any `category_keep` flag is
   forced in even if it missed its threshold, and survives the drop list.
3. **Category drop (blacklist)** — a gene matching any `category_drop` flag is
   removed (default: `is_pseudogene`).

Flags come from the Ensembl GTF reference (`featuresel.py ref`):
`is_protein_coding`, `is_pseudogene`, `is_OR`, `is_vomeronasal`, `is_taste`,
`is_IG_{V,D,J,C}`, `is_TR_{V,D,J,C}`, `is_mt`, `is_hb`, `is_ribo`, `is_sex`.

## Tuning the thresholds

Measure once, then retune for free — thresholds and category lists are build-time
only. Every snapshot ships `thresholds_{sp}.tsv`: gene counts per biotype at HVG
support >= 1, 2, 3, 5, 10, 20, 50. Read a threshold off that table, then

```bash
featuresel.py build --hvg-min-datasets lncRNA=15,miRNA=5 --tag t2
```

which merges onto the config rules (a bare `--hvg-min-datasets 5` sets a flat one).

`votes.py` renders the same decision interactively: `cache/figs/votes_{sp}.html`
has sliders for `s_min` (grid 1.0–3.0) and the vote cutoff, per-biotype and
per-flag tables of passing genes, the vote-count histogram of any category and
the passing gene list. Votes are precomputed per gene on the `s_min` grid, so the
page is a single offline file.

## Expression statistics

Descriptive only — nothing filters on them, they are there to judge whether a
selected gene is actually expressed. Per gene, pooled over all datasets where it
is present:

| column | meaning |
|---|---|
| `n_datasets_present` | datasets containing the gene |
| `n_datasets_hvg` / `best_hvg_rank` | HVG support and best per-dataset rank |
| `pooled_det` | detected cells / total cells |
| `max_det`, `median_det` | per-dataset detection rate, max and median |
| `mean_counts_per_cell` | total counts / total cells |
| `mean_counts_in_positive` | total counts / detected cells |

## Numerical guards

Not QC (the rsi inputs are QC'd): cells with < `min_cell_counts` (10) total
counts are dropped, and per group a gene seen in < `min_gene_cells` (3) cells is
not fitted or ranked. Genes seen in one cell with one count all share the same
(mean, var); hundreds of identical points made skmisc's loess singular — on
mca3.0 Brain this guard took the fallbacks from 6 lineages to 1. scanpy's own
Pearson-residual call, which has no such guard, picks 231 of its top-2000 from
exactly those genes.

## Cost

Three streaming passes over stored nonzeros (depth, moments, standardized
variance), one disk read (the Reader caches the first pass in RAM), never an
AnnData. mca3.0 Brain (28472 x 39272, 16.7M nnz): 4.2 s, of which the loess fits
for 11 groups are 0.2 s — the fit everyone worries about is free. Dense-stored
files are ~97% zeros; converting them to stored entries on read takes a 1.5 GB
matrix from 22 s to 0.5 s. With 5 local processes the 160-dataset mouse corpus
(1.6M cells, 115 GB) measures in a few minutes; a Slurm array is slower in
practice because of queueing. Pearson residuals need dense O(cells x genes)
work (125 s on Brain) and are kept only for comparison.

## Caching

Per-dataset measurement (`cache/stats/<sample_key>.parquet`) holds one row per
harmonized gene. It is recomputed only when the h5ad is newer or the `hvg`
config block changes (hashed into each meta JSON). `build` is instant and
re-runnable: it only aggregates and applies category rules.
