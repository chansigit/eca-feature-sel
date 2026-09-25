# eca-feature-sel

**Pick the gene vocabulary for a single-cell foundation model.**

Given a corpus of harmonized `h5ad` files, this tool finds the genes that are
*highly variable* in each dataset, lets every dataset cast one vote per gene,
and keeps the genes that collect enough votes for their category. The result is
a token list that is driven by the data (a gene is in because it carries
cell-state information somewhere in the corpus), reproducible, and cheap to
re-tune.

Current mouse vocabulary: **18,794 genes** from 166 datasets / 2.5 M cells
(18,216 protein-coding, 557 lncRNA, 21 immunoglobulin / TCR constant genes).

---

## Quick start

```bash
# 1. point config.yaml at your inputs, a cache dir on fast storage, and a python
#    with: numpy pandas pyarrow scipy h5py scikit-misc pyyaml  (matplotlib for reports)
# 2. run everything
./pipeline.sh                 # scan -> reference -> measure -> vote -> explorer

# the deliverable
cache/vocab/final/genes_mouse.tsv
```

`pipeline.sh` is incremental: rerun it whenever new datasets land and only the
new or changed files are measured. 160 datasets take about three minutes on
five CPUs; one 400 k-cell dataset takes about two minutes.

---

## How it works

```
 per dataset (cached)                     across the corpus (instant)
 ┌──────────────────────────────┐        ┌──────────────────────────────┐
 │ stream raw counts            │        │ one dataset = one vote       │
 │ score every gene:            │        │ protein_coding  ≥ 3 votes    │
 │   whole dataset              │  ───►  │ everything else ≥ 10 votes   │
 │   + each coarse lineage      │        │ then keep / drop lists       │
 │ store score + rank per group │        │ → genes_<species>.tsv        │
 └──────────────────────────────┘        └──────────────────────────────┘
```

**1. Score.** For each dataset the worker streams `layers/counts` with h5py
(no AnnData, one pass over the file, the rest replayed from RAM) and computes
the Seurat v3 "vst" standardized variance for every gene: fit a loess trend of
variance against mean, then measure how far each gene sits above the trend.
A score of 1 means "as variable as a typical gene at that expression level";
Apoe in liver scores 40. Scores are comparable across datasets, so there is
no fixed "top 2000".

**2. Lineages too.** The same scoring is repeated inside each coarse lineage
(`obs.zmip_ann_coarse` or similar, at least 200 cells), because the dataset-wide
call is dominated by the majority population and misses genes that only vary
inside, say, the T cells. A dataset's HVG set is the union.

**3. Vote.** A gene passes in a group when `score >= s_min` (1.3) and
`rank <= n_max` (3000). A dataset votes for a gene if it passes in the whole
dataset or in any of its lineages. Votes are counted, not weighted: a
low-depth dataset simply passes fewer genes and contributes less.

**4. Category rules.** Protein-coding genes need 3 votes, every other biotype
10. Then a whitelist and a blacklist are applied, tuned for a foundation-model
vocabulary:

| | categories | reason |
|---|---|---|
| keep | IG / TR constant genes, sex genes | cell-state markers even when few datasets vote |
| drop | pseudogenes | multi-mapping artefacts |
| drop | IG / TR V, D, J segments | clonotype identity, not cell state |
| drop | miRNA, snoRNA, snRNA, scaRNA, misc_RNA, rRNA, Mt_tRNA, Mt_rRNA | poly-A capture artefacts |
| drop | TEC | unconfirmed loci |

Mitochondrial, ribosomal, hemoglobin and olfactory-receptor genes are *not*
blanket-dropped; they pass or fail on votes like any other gene.

Only two things are ever "cleaned": cells with fewer than 10 counts and, per
group, genes seen in fewer than 3 cells. Those are numerical guards for the
loess fit, not quality control.

---

## Outputs

Every `build` writes a snapshot to `cache/vocab/<tag>/` and points
`cache/vocab/latest` at it.

| file | what |
|---|---|
| `genes_<sp>.tsv` | **the vocabulary**: `harmonized_id`, `symbol`, `biotype`, `n_datasets_hvg` |
| `vocab_<sp>.tsv` | every gene seen in the corpus with `selected`, vote counts, best / median score, detection rate, mean counts, category flags |
| `thresholds_<sp>.tsv` | genes per biotype at vote thresholds 1, 2, 3, 5, 10, 20, 50 |
| `sweep_smin_<sp>.tsv` | the whole selection re-run at s_min 1.1 … 2.0 |
| `params.json` | every parameter that produced the snapshot |

Per-dataset measurements live in `cache/stats/<sample_key>.parquet` (gene
stats, dataset-level score and rank), `.lineages.parquet` (top genes per
lineage) and `.meta.json` (groups, timings, loess fallbacks).

---

## Looking at the data

```bash
python votes.py                         # cache/figs/votes_<sp>.html
python report.py --all                  # cache/reports/index.html
python explore.py <sample_key>          # cache/figs/hvg_explorer_<sample_key>.html
```

- **Corpus vote explorer** (`votes.py`): one offline page with sliders for
  `s_min` and the vote cutoff; tables of passing genes per biotype and per
  flag, the vote histogram of any category, and the gene list behind it. Use it
  to choose thresholds before you build.
- **Per-dataset reports** (`report.py`): mean-variance plot with the loess
  trend, score distribution, which lineages contributed, top voted genes.
- **Single-dataset explorer** (`explore.py`): interactive version of the report
  with vst vs. Pearson-residual scoring and the legacy dispersion method side by
  side.

---

## Configuration

Everything lives in `config.yaml`. The knobs you will actually touch:

```yaml
inputs_tsv: [mouse.tsv]         # sample_key <tab> species <tab> h5ad
cache_root: /scratch/.../cache  # fast storage, never $HOME

hvg:                            # measurement (changing these re-measures)
  min_lineage_cells: 200
  lineage_obs_key: [zmip_ann_coarse, msp_ann_coarse, cell_lineage]

selection:                      # build time (free to change)
  s_min: 1.3
  n_max: 3000
  hvg_min_datasets: {protein_coding: 3, default: 10}
  category_keep: [is_IG_C, is_TR_C, is_sex]
  category_drop: [is_pseudogene, is_IG_V, is_IG_D, is_IG_J, is_TR_V, is_TR_D, is_TR_J,
                  TEC, miRNA, snoRNA, snRNA, scaRNA, misc_RNA, rRNA, Mt_tRNA, Mt_rRNA, ribozyme]
```

Keys in the rules and lists are `is_*` flags (`is_protein_coding`,
`is_pseudogene`, `is_OR`, `is_vomeronasal`, `is_taste`, `is_mt`, `is_hb`,
`is_ribo`, `is_sex`, `is_IG_{V,D,J,C}`, `is_TR_{V,D,J,C}`) or exact Ensembl
biotypes. Anything under `selection:` can also be overridden per build:

```bash
python featuresel.py build --s-min 1.5 --tag strict
python featuresel.py build --hvg-min-datasets protein_coding=5,lncRNA=20 --tag t2
python featuresel.py build --category-drop is_pseudogene,is_OR --tag noOR
python featuresel.py build --hvg-source global --tag nolineage      # ignore lineage votes
```

---

## Step by step

```bash
python scan_rsi.py                 # eca-rsi results -> mouse.tsv (--species human, --list)
python featuresel.py status        # what is measured, stale, missing
python featuresel.py ref           # biotype reference: Ensembl GTF + MGI feature types
python featuresel.py measure --local   # this node, all CPUs (--jobs N, --force)
python featuresel.py measure       # or one Slurm array task per dataset
python featuresel.py build --tag v1
python featuresel.py refresh       # measure, wait, build
```

A dataset is re-measured only if its `h5ad` is newer than the cached result or
the `hvg:` block of the config changed (its hash is stored with each result).
Removing a dataset from the TSV removes it from the next build.

### Where inputs come from

`scan_rsi.py` walks two kinds of roots (`--root`, repeatable): the Oak corpus
(`<dataset>/.../rsi/units/<unit>/release/final.h5ad`) and the gen2 run batches on
scratch (`<batch>/<run>/units/<unit>/release/final.h5ad`, keyed by the run's
`dataset_id` from `spec.json`). A gen2 run whose `input_root` points into an Oak
dataset supersedes that dataset's Oak run, so nothing is counted twice; the same
dataset finished twice in a batch keeps the newest run. Keys in `exclude.txt`
are left out by hand.

### Input requirements

Each `h5ad` needs `layers/counts` (raw counts; csr, csc or dense all work) and
`var.gene_id_harmonized` (Ensembl ids; MGI accessions are accepted for mouse
genes without one). Rows without a harmonized id are dropped; duplicate ids are
summed. A coarse lineage column in `obs` is optional but recommended.

The input TSV has three tab-separated columns, `sample_key`, `species`,
`h5ad`; `#` comments and a header row are allowed, relative paths resolve
against the TSV.

---

## Repository layout

```
pipeline.sh      scan -> ref -> measure -> build -> votes (incremental)
featuresel.py    CLI: status / measure / ref / build / refresh
worker.py        one dataset: streaming stats + scores -> parquet
hvg.py           the scoring: vst (and Pearson residuals), dataset-level + per lineage
votes.py         corpus vote explorer (HTML)
report.py        per-dataset HTML reports
explore.py       single-dataset interactive explorer (HTML)
scan_rsi.py      eca-rsi results (Oak corpus + gen2 batches on scratch) -> input TSV
exclude.txt      sample_keys to leave out of the TSVs, with reasons
test_worker.py   self-check (python test_worker.py)
config.yaml      paths, Slurm resources, hvg / selection defaults
mouse.tsv        mouse inputs        human.tsv   human inputs
docs/            selection_design.md: rules, flags, stat columns, costs
```

---

## Notes

- Verified against scanpy: `seurat_v3` Jaccard > 0.95 on top-N, Pearson
  residuals to 1e-13. Pearson residuals (`hvg.method: pearson`) are kept for
  comparison only; they are dense and ~50x slower.
- The file's own `var.highly_variable` is carried along as `hvg_upstream`
  (`--hvg-source upstream`) so you can compare with whatever the upstream
  pipeline chose.
- Species are handled independently in their own id spaces; there is no
  ortholog mapping.
- The whole thing is I/O bound. On a shared filesystem the streaming read runs
  at the node's bandwidth; more CPUs help linearly, a faster language would not.
- **Slow filesystem days.** Every filesystem call (directory scan, stat, open)
  runs in a forked child with a deadline (`scan_rsi.py --timeout`, default 120 s;
  `measure --timeout` per dataset, default 1800 s). Whatever does not answer is
  *deferred*: the scan keeps that dataset's row from the previous TSV, `status`
  and `build` use its cached measurement, and the next run checks it again.
  Nothing is ever dropped because a mount hung.
