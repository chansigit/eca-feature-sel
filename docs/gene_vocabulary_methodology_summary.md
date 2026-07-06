# Methodology: Gene Vocabulary Selection

## Input

We start from an explicit dataset list for each species. Each dataset contributes raw gene-count measurements and harmonized gene identifiers. Features without a valid harmonized gene identifier are excluded.

If multiple features map to the same harmonized gene, they are treated as one gene for detection. This prevents duplicate feature rows from inflating detection rates.

## Vocabulary Rules

For each gene, detection rate measures the fraction of cells in which that gene has nonzero counts.

Gene vocabulary selection is controlled by three inclusion rules and one category rule.

The detection rule selects genes that are detected in at least a minimum fraction of cells in at least a minimum number of datasets. The default threshold is detection in at least 1% of cells in at least 10 datasets.

The HVG rule selects highly variable genes independently within each dataset. The default setting selects the top 3,000 highly variable genes per dataset.

The cluster-DEG rule selects genes that distinguish a cell-type or cluster from the remaining cells within the same dataset. Each eligible group is compared against all other cells in that dataset, and top marker genes are retained after minimum group size, expression prevalence, effect-size, and significance filters.

The category rule removes genes from excluded gene categories. The current category rule excludes pseudogenes.

A gene is included in the final vocabulary if it passes the detection rule, the HVG rule, or the cluster-DEG rule, and is not removed by the category rule.

## Output

The output is one vocabulary per species, indexed by harmonized gene identifiers. The vocabulary size is determined by the data and threshold settings, not fixed in advance.
