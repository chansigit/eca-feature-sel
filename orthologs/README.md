# Human–mouse ortholog core

The part of the two released vocabularies that maps across species **and** was
selected on both sides: an ortholog pair (h, m) is kept when h is in the human
vocabulary and m in the mouse vocabulary.

```
python orthologs/ortholog_vocab.py --human human-v1 --mouse mouse-v2
```

Ortholog table: `pairs.parquet`, copied from
`/scratch/users/chensj16/projects/genofoundation/data/orthologs/v111/pairs.parquet`
(Ensembl 111 homologies, five species; only the human–mouse rows are used). It is
not in git (1.4 MB, regenerated upstream); copy it here before running.

## Result (human-v1 × mouse-v2)

| | human | mouse |
|---|---|---|
| vocabulary | 20,956 | 19,131 |
| **kept (has a partner in the other vocabulary)** | **15,630** | **15,666** |
| no ortholog in the table | 5,092 | 3,162 |
| ortholog exists, but none of them is in the other vocabulary | 227 | 238 |
| no Ensembl id (HGNC / MGI accession), not in the table | 7 | 65 |

15,805 kept pairs: 15,118 one-to-one, 540 one-to-many, 147 many-to-many.
Of the unmapped genes, lncRNAs are almost all of the human non-protein-coding
loss (2,741) and every mouse lncRNA (689) is unmapped: the table has few
lncRNA orthologs.

## Files

| file | what |
|---|---|
| `shared_pairs.tsv` | kept pairs: `human_id`, `human_symbol`, `mouse_id`, `mouse_symbol`, `homology_type`, `confidence` |
| `shared_human.tsv` / `shared_mouse.tsv` | the kept genes on each side, one row per gene (vocabulary columns) |
| `unmapped_human.tsv` / `unmapped_mouse.tsv` | the rest, with `why_unmapped` |
| `summary.json` | the counts above |
