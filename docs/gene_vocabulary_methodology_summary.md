# Methodology: Gene Vocabulary Selection

## Input

We define an explicit dataset list for each species. Each dataset contributes a raw count matrix and a harmonized gene identifier for each feature. Features without a valid harmonized gene identifier are excluded.

For dataset d, let X_d be the raw count matrix with n_d cells. Let h_d(j) be the harmonized gene assigned to feature j. For gene g, let V_d(g) = {j : h_d(j) = g}.

## Per-Dataset Statistics

For each dataset d and gene g, we compute:

sum_counts(d, g) = ∑_{i=1}^{n_d} ∑_{j∈V_d(g)} X_d[i, j]

n_detected(d, g) = ∑_{i=1}^{n_d} 1{∑_{j∈V_d(g)} X_d[i, j] > 0}

detection_rate(d, g) = n_detected(d, g) / n_d

If multiple features map to the same harmonized gene, detection is counted once per cell. This prevents duplicate feature rows from inflating detection rates.

## Cross-Dataset Summary

For each species, gene statistics are aggregated across its datasets. For each gene, we summarize its dataset coverage, maximum detection rate, median detection rate, pooled detection rate, and maximum expression among detected cells.

pooled_detection_rate(g) = ∑_d n_detected(d, g) / ∑_d n_d

## Vocabulary Rule

A gene is included if it satisfies either a consistency rule or a strength rule.

The consistency rule keeps genes detected above threshold f in at least k datasets:

n_datasets_with_detection_rate_above_f(g) ≥ k

The strength rule keeps strong context-specific genes:

max_detection_rate(g) ≥ τ_detection and max_mean_expr(g) ≥ Q_q

Here, Q_q is the q-th quantile of max_mean_expr among genes passing the consistency rule.

The final candidate set is:

Candidate(g) = Consistency(g) OR Strength(g)

## Optional Veto

Biotype and gene-family filters are optional vetoes applied after candidate selection. They do not determine primary inclusion.

## Output

The output is one vocabulary per species, indexed by harmonized gene identifiers. The vocabulary size is determined by the data and threshold settings, not fixed in advance.
