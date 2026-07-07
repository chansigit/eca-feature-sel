# Selection Design

## Goal

Gene inclusion is driven by three sources of evidence: cross-dataset detection, dataset-level HVG selection, and within-dataset cluster markers. The category rule is applied after these inclusion rules.

## Inclusion Rules

The detection rule keeps broadly observed genes. A gene passes this rule when its detection rate is at least `min_frac` in at least `min_dataset_occurrence` datasets. The defaults are `min_frac = 0.01` and `min_dataset_occurrence = 10`.

The HVG rule keeps genes selected as highly variable within individual datasets. Each dataset contributes its top `n_top_genes` HVGs, with a default of `n_top_genes = 3000`. A gene passes the HVG rule if it is selected in at least `min_datasets` datasets.

The cluster-DEG rule keeps cell-type or state marker genes. Within each dataset, eligible cell-type or cluster groups are compared against the remaining cells using a one-vs-rest test. A gene passes the rule if it is retained as a marker in at least `min_datasets` datasets.

## Default Hyperparameters

The design defaults are recorded in `config.yaml` under `selection`.

Detection defaults are `min_frac = 0.01` and `min_dataset_occurrence = 10`.

HVG defaults are `n_top_genes = 3000`, `min_datasets = 1`, `layer = counts`, `flavor = seurat_v3`, and `normalize_target_sum = 10000`.

Cluster-DEG defaults use the first available observation key from `cell_type_fine`, `cell_type_coarse`, `cell_type`, `cell_subtype`, `cluster_annotation`, `cluster`, and `clusters`. Groups with fewer than `50` cells are skipped. Marker filters are `min_pct_in_cluster = 0.10`, `min_pct_difference = 0.05`, `min_log2fc = 0.25`, `max_adj_pvalue = 0.05`, and `top_n_per_cluster = 100`. The DEG worker first inspects `top_n_per_cluster * candidate_multiplier` ranked candidates per group before applying these filters.

The category rule currently removes pseudogenes.

## Output Columns

The final vocabulary table reports which rule selected each gene. Key columns are `selected_by_detection`, `selected_by_hvg`, `selected_by_cluster_deg`, `category_excluded`, and `selected`.

The table also reports support counts, including `n_datasets_detection`, `n_datasets_hvg`, and `n_datasets_cluster_deg`.

## Implementation Notes

HVG and cluster-DEG statistics should be cached per dataset, so changing selection thresholds does not require re-reading all input files.
