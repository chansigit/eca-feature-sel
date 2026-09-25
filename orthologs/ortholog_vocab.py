"""Human-mouse ortholog core of the two released vocabularies.

Keeps the ortholog pairs (h, m) where h is in the human vocabulary and m in the
mouse vocabulary, i.e. genes that map across AND whose partner was selected on
the other side. Everything else is counted and listed.

    python orthologs/ortholog_vocab.py [--human TAG] [--mouse TAG]

Inputs:  cache/vocab/<tag>/genes_{human,mouse}.tsv, orthologs/pairs.parquet
         (Ensembl 111 homologies; human-mouse rows only are used).
Outputs (orthologs/):
  shared_pairs.tsv     kept pairs: human id/symbol, mouse id/symbol, homology_type, confidence
  shared_human.tsv     human vocabulary genes with a kept partner (one row per gene)
  shared_mouse.tsv     mouse vocabulary genes with a kept partner
  unmapped_human.tsv   human vocabulary genes without a kept partner + why
  unmapped_mouse.tsv   mouse vocabulary genes without a kept partner + why
  summary.json         the counts
"""
import argparse
import json
import os

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", default="human-v1", help="vocab snapshot tag")
    ap.add_argument("--mouse", default="mouse-v2", help="vocab snapshot tag")
    ap.add_argument("--pairs", default=os.path.join(HERE, "pairs.parquet"))
    a = ap.parse_args()
    vocab = os.path.join(yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))["cache_root"], "vocab")
    H = pd.read_csv(f"{vocab}/{a.human}/genes_human.tsv", sep="\t")
    M = pd.read_csv(f"{vocab}/{a.mouse}/genes_mouse.tsv", sep="\t")

    p = pd.read_parquet(a.pairs)
    hm = p[(p.species_a == "homo_sapiens") & (p.species_b == "mus_musculus")]
    mh = p[(p.species_a == "mus_musculus") & (p.species_b == "homo_sapiens")]
    pairs = pd.concat([hm.rename(columns={"gene_a": "human_id", "gene_b": "mouse_id"}),
                       mh.rename(columns={"gene_a": "mouse_id", "gene_b": "human_id"})])
    pairs = (pairs[["human_id", "mouse_id", "homology_type", "confidence"]]
             .drop_duplicates(["human_id", "mouse_id"]))

    hset, mset = set(H.harmonized_id), set(M.harmonized_id)
    kept = pairs[pairs.human_id.isin(hset) & pairs.mouse_id.isin(mset)].copy()
    kept = (kept.merge(H[["harmonized_id", "symbol"]].rename(columns={"harmonized_id": "human_id", "symbol": "human_symbol"}))
                .merge(M[["harmonized_id", "symbol"]].rename(columns={"harmonized_id": "mouse_id", "symbol": "mouse_symbol"}))
                [["human_id", "human_symbol", "mouse_id", "mouse_symbol", "homology_type", "confidence"]]
                .sort_values(["human_symbol", "mouse_symbol"]))

    def classify(V, col, other_col, other_set):
        """Why a vocabulary gene has no kept partner."""
        has_any = set(pairs[col])
        partner_in = set(pairs.loc[pairs[other_col].isin(other_set), col])
        why = []
        for g in V.harmonized_id:
            if not g.startswith("ENS"):
                why.append("no Ensembl id (MGI/HGNC accession): not in the ortholog table")
            elif g not in has_any:
                why.append("no ortholog in the table")
            elif g not in partner_in:
                why.append("ortholog(s) exist but none is in the other vocabulary")
            else:
                why.append("")
        return pd.Series(why, index=V.index)

    H["why_unmapped"] = classify(H, "human_id", "mouse_id", mset)
    M["why_unmapped"] = classify(M, "mouse_id", "human_id", hset)
    shared_h = H[H.why_unmapped == ""].drop(columns="why_unmapped")
    shared_m = M[M.why_unmapped == ""].drop(columns="why_unmapped")
    un_h = H[H.why_unmapped != ""]
    un_m = M[M.why_unmapped != ""]

    kept.to_csv(os.path.join(HERE, "shared_pairs.tsv"), sep="\t", index=False)
    shared_h.to_csv(os.path.join(HERE, "shared_human.tsv"), sep="\t", index=False)
    shared_m.to_csv(os.path.join(HERE, "shared_mouse.tsv"), sep="\t", index=False)
    un_h.to_csv(os.path.join(HERE, "unmapped_human.tsv"), sep="\t", index=False)
    un_m.to_csv(os.path.join(HERE, "unmapped_mouse.tsv"), sep="\t", index=False)

    one2one = kept[kept.homology_type == "ortholog_one2one"]
    summary = {
        "human_vocab": a.human, "mouse_vocab": a.mouse,
        "human_genes": len(H), "mouse_genes": len(M),
        "human_mouse_pairs_in_table": len(pairs),
        "kept_pairs": len(kept),
        "kept_pairs_by_type": kept.homology_type.value_counts().to_dict(),
        "shared_human_genes": len(shared_h), "shared_mouse_genes": len(shared_m),
        "shared_one2one_genes": int(one2one.human_id.nunique()),
        "unmapped_human": un_h.why_unmapped.value_counts().to_dict(),
        "unmapped_mouse": un_m.why_unmapped.value_counts().to_dict(),
        "unmapped_human_by_biotype": un_h.biotype.fillna("(none)").value_counts().to_dict(),
        "unmapped_mouse_by_biotype": un_m.biotype.fillna("(none)").value_counts().to_dict(),
    }
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
