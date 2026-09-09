#!/usr/bin/env python
"""Per-dataset report from the cached measurement (no recomputation).

    report.py mca3.0-brain-mouse            # -> cache/reports/mca3.0-brain-mouse.html (+ .genes.tsv)
    report.py --all                         # every measured dataset + index.html

Selection thresholds (s_min, n_max, min_lineage_cells, min_lineages) come from
config `selection:` and can be overridden; the report shows what THIS dataset
would vote for under them.
"""
import argparse
import base64
import html
import io
import json
import os

import numpy as np
import pandas as pd

import featuresel

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ figures
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figures(main, lin, meta, sel, s_min, n_max, names, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = (main["rank"] > 0) & main["score"].notna()
    m, v, sc = main.loc[ok, "sum_counts"] / main.loc[ok, "n_cells"], main.loc[ok, "var"], main.loc[ok, "score"]
    pas = sel["global"].reindex(main.index).fillna(False).astype(bool)[ok]
    out = {}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    a.scatter(np.log10(m[~pas]), np.log10(v[~pas]), s=2, c="#bbb", label="other genes")
    a.scatter(np.log10(m[pas]), np.log10(v[pas]), s=4, c="#d62728", label=f"pass (score≥{s_min}, rank≤{n_max}): {int(pas.sum())}")
    ev = main.loc[ok, "expected_var"]
    o = np.argsort(m.to_numpy())
    a.plot(np.log10(m.to_numpy()[o]), np.log10(ev.to_numpy()[o]), c="k", lw=1.5, label="loess trend (expected var)")
    a.set_xlabel("log10 mean (raw counts)"); a.set_ylabel("log10 variance"); a.set_title(f"A. mean–variance, all cells (N={sizes[0]})")
    a.legend(markerscale=4, fontsize=9)
    a = ax[1]
    a.scatter(np.log10(m[~pas]), sc[~pas], s=2, c="#bbb"); a.scatter(np.log10(m[pas]), sc[pas], s=4, c="#d62728")
    a.axhline(1, c="k", ls="--", lw=1, label="score = 1 (on trend)"); a.axhline(s_min, c="#1f77b4", ls="--", lw=1, label=f"s_min = {s_min}")
    a.set_yscale("log"); a.set_xlabel("log10 mean"); a.set_ylabel("score (standardized variance)"); a.set_title("B. score vs expression level"); a.legend(fontsize=9)
    out["ab"] = _png(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    curve = np.sort(sc.to_numpy())[::-1]
    a.plot(np.arange(1, len(curve) + 1), curve, c=cols[0], lw=1.6, label=f"all cells (N={sizes[0]})")
    for i, (name, size) in enumerate(zip(names[1:], sizes[1:]), start=1):
        s = lin.loc[(lin["lineage"] == name) & (lin["rank"] > 0), "score"].sort_values(ascending=False).to_numpy()
        if len(s):
            a.plot(np.arange(1, len(s) + 1), s, c=cols[i % 10], lw=1.1, label=f"{name} (N={size})")
    a.axvline(n_max, c="k", ls=":", lw=1); a.axhline(s_min, c="#1f77b4", ls="--", lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("rank within group"); a.set_ylabel("score")
    a.set_title("C. rank–score per group (dotted: n_max, dashed: s_min)"); a.legend(fontsize=7, ncol=2)
    a = ax[1]
    counts = [int(sel["global"].sum())] + [int(sel["per_lineage"].get(n, 0)) for n in names[1:]]
    a.bar(range(len(names)), counts, color=["#1f77b4"] + ["#ff7f0e"] * (len(names) - 1))
    a.set_xticks(range(len(names))); a.set_xticklabels([f"{n}\nN={s}" for n, s in zip(names, sizes)], rotation=60, ha="right", fontsize=8)
    a.set_ylabel("# genes passing"); a.set_title("D. genes passing per group")
    out["cd"] = _png(fig)
    return out


# ------------------------------------------------------------------ selection
def select(main, lin, s_min, n_max, min_cells, min_lineages):
    """What this dataset votes for, under the given thresholds."""
    g = featuresel._passes(main, s_min, n_max)
    lin_ok = lin[(lin["lineage_cells"] >= min_cells) & featuresel._passes(lin, s_min, n_max)]
    per_gene = lin_ok.groupby("harmonized_id")["lineage"].agg(["nunique", lambda x: ", ".join(sorted(x))])
    per_gene.columns = ["n_lineages", "lineages"]
    via_lin = per_gene[per_gene["n_lineages"] >= min_lineages]
    votes = pd.Index(main.index[g]).union(via_lin.index)
    return {"global": g, "lineage_genes": via_lin, "votes": votes,
            "per_lineage": lin_ok.groupby("lineage").size().to_dict(),
            "skipped_lineages": sorted(set(lin["lineage"]) - set(lin_ok["lineage"]))}


# ------------------------------------------------------------------ html
CSS = """body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
header{padding:14px 22px;background:#1f2937;color:#fff}header h1{margin:0;font-size:18px}header .sub{opacity:.8;font-size:13px;margin-top:4px}
.wrap{padding:16px 22px;max-width:1300px;margin:auto}.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;margin-bottom:16px}
.card h2{margin:0 0 8px;font-size:15px}.kpi{display:flex;gap:28px;flex-wrap:wrap;margin:6px 0 10px}.kpi div{font-size:13px;color:#555}.kpi div b{display:block;font-size:22px;color:#111}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f3f4f6}
.mut{color:#888}.warn{color:#b45309}img{max-width:100%}.scroll{max-height:420px;overflow:auto}code{background:#f3f4f6;padding:0 4px;border-radius:3px}"""


def esc(x):
    return html.escape(str(x))


def render(key, main, lin, meta, sel, figs, s_min, n_max, min_cells, min_lineages, ref):
    names, sizes = meta["groups"], meta["group_sizes"]
    up = meta.get("upstream_hvg") or {}
    votes = sel["votes"]
    g_only = int(sel["global"].sum() - len(set(main.index[sel["global"]]) & set(sel["lineage_genes"].index)))
    l_only = len(set(sel["lineage_genes"].index) - set(main.index[sel["global"]]))
    up_ov = int(main.loc[votes, "hvg_upstream"].sum()) if up else None

    def biotype(ids):
        if ref is None:
            return "—"
        b = ref.reindex(ids)["biotype"].fillna("(none)").value_counts()
        return ", ".join(f"{k} {v}" for k, v in b.head(5).items())

    rows = []
    for i, (n, s) in enumerate(zip(names, sizes)):
        note = "; ".join(meta.get("fit_notes", {}).get(n, [])) or ("loess" if i == 0 or n in sel["per_lineage"] else "")
        passing = int(sel["global"].sum()) if i == 0 else sel["per_lineage"].get(n, 0)
        skipped = i > 0 and s < min_cells
        sar = meta.get("score_at_rank", {})
        rows.append(f"<tr class='{'mut' if skipped else ''}'><td>{esc(n)}{' (below min_lineage_cells)' if skipped else ''}</td>"
                    f"<td>{s}</td><td>{meta['n_rankable'][i]}</td><td><b>{passing if not skipped else '–'}</b></td>"
                    f"<td>{_fmt(sar.get('500', [None]*len(names))[i])}</td><td>{_fmt(sar.get('2000', [None]*len(names))[i])}</td>"
                    f"<td class='{'warn' if 'failed' in note else ''}'>{esc(note)}</td></tr>")

    top = main.loc[votes].copy()
    top["via"] = np.where(sel["global"].reindex(top.index).fillna(False), "global", "")
    lg = sel["lineage_genes"].reindex(top.index)
    top["via"] = top["via"].where(lg["n_lineages"].isna(), top["via"] + np.where(top["via"] != "", " + ", "") + "lineage(" + lg["n_lineages"].fillna(0).astype(int).astype(str) + ")")
    top["lineages"] = lg["lineages"].fillna("")
    top["mean"] = top["sum_counts"] / top["n_cells"]
    top["det"] = top["n_detected"] / top["n_cells"]
    if ref is not None:
        top["biotype"] = ref.reindex(top.index)["biotype"].fillna("")
    top = top.sort_values("score", ascending=False)
    cols = ["symbol", "score", "rank", "mean", "det", "via", "lineages"] + (["biotype"] if ref is not None else [])
    trows = "".join("<tr>" + "".join(f"<td>{_fmt(r[c]) if c in ('score','mean','det') else esc(r[c])}</td>" for c in cols) + "</tr>"
                    for _, r in top.head(40).iterrows())

    lin_only = top[(top["via"].str.startswith("lineage"))].head(25)
    lrows = "".join(f"<tr><td>{esc(r.symbol)}</td><td>{_fmt(r.score)}</td><td>{r['rank'] or '–'}</td><td>{esc(r.lineages)}</td>"
                    f"<td>{_fmt(r['mean'])}</td><td>{_fmt(r.det)}</td></tr>" for _, r in lin_only.iterrows())

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>HVG report — {esc(key)}</title><style>{CSS}</style></head><body>
<header><h1>HVG report — {esc(key)}</h1><div class="sub">{esc(meta.get('species',''))} · {esc(os.path.basename(meta['h5ad']))} · method {esc(meta.get('method','vst'))} · thresholds: s_min {s_min}, n_max {n_max}, min_lineage_cells {min_cells}, min_lineages {min_lineages}</div></header>
<div class="wrap">
<div class="card"><h2>Dataset</h2>
<div class="kpi"><div>cells used<b>{sizes[0]:,}</b></div><div>cells dropped (&lt;{esc(meta.get('min_cell_counts', 10))} counts)<b>{meta.get('n_cells_dropped', 0)}</b></div>
<div>harmonized genes<b>{len(main):,}</b></div><div>unmapped var rows<b>{meta.get('n_unmapped_rows', 0)}</b></div><div>duplicate ids<b>{meta.get('n_duplicate_harmonized_ids', 0)}</b></div>
<div>encoding<b style="font-size:16px">{esc(meta.get('encoding',''))}</b></div><div>lineage column<b style="font-size:16px">{esc(meta.get('lineage_obs_key') or 'none')}</b></div>
<div>lineages kept / total<b>{meta.get('lineages_kept', 0)} / {meta.get('lineages_total', 0)}</b></div><div>HVG compute<b style="font-size:16px">{meta.get('hvg_seconds','?')} s</b></div></div></div>

<div class="card"><h2>What this dataset votes for</h2>
<div class="kpi"><div>genes voted<b>{len(votes):,}</b></div><div>via dataset-level call<b>{int(sel['global'].sum()):,}</b></div><div>only via lineages<b>{l_only:,}</b></div><div>only via dataset-level<b>{g_only:,}</b></div>
<div>overlap with upstream rsi HVG ({up.get('n','–')}, {esc(up.get('flavor','–'))})<b>{up_ov if up_ov is not None else '–'}</b></div></div>
<p class="mut">biotypes of voted genes: {esc(biotype(votes))}</p>
<p class="mut">Full list: <code>{esc(key)}.genes.tsv</code> next to this file.</p></div>

<div class="card"><h2>Groups</h2><table><tr><th>group</th><th>cells</th><th>rankable genes</th><th>passing</th><th>score @ rank 500</th><th>score @ rank 2000</th><th>trend fit</th></tr>{''.join(rows)}</table>
{('<p class="warn">lineages present but below min_lineage_cells: ' + esc(', '.join(sel['skipped_lineages'])) + '</p>') if sel['skipped_lineages'] else ''}</div>

<div class="card"><h2>Figures</h2><img src="{figs['ab']}"><img src="{figs['cd']}"></div>

<div class="card"><h2>Top 40 voted genes (by dataset-level score)</h2><div class="scroll"><table><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr>{trows}</table></div></div>

<div class="card"><h2>Genes the lineage split added (top 25, absent from the dataset-level call)</h2>
<div class="scroll"><table><tr><th>symbol</th><th>best lineage score</th><th>dataset-level rank</th><th>lineages</th><th>mean</th><th>det</th></tr>{lrows}</table></div>
<p class="mut">"dataset-level rank" is where the whole-dataset call put the gene; these are the ones a fixed global top-N would have missed.</p></div>
</div></body></html>"""


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.3g}" if isinstance(x, (float, np.floating)) else str(x)


# ------------------------------------------------------------------ driver
def build_report(cfg, rec, outdir, s_min, n_max, min_cells, min_lineages, ref):
    main = pd.read_parquet(rec["out"]).set_index("harmonized_id")
    lin = pd.read_parquet(os.path.splitext(rec["out"])[0] + ".lineages.parquet")
    meta = json.load(open(os.path.splitext(rec["out"])[0] + ".meta.json"))
    # lineage score for the "added" table: best lineage score per gene
    sel = select(main, lin, s_min, n_max, min_cells, min_lineages)
    best_lin = lin.groupby("harmonized_id")["score"].max()
    main["score"] = main["score"].where(main["score"].notna(), np.nan)
    figs = figures(main, lin, meta, sel, s_min, n_max, meta["groups"], meta["group_sizes"])
    page = render(rec["key"], main, lin, meta, sel, figs, s_min, n_max, min_cells, min_lineages, ref)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{rec['key']}.html"), "w") as fh:
        fh.write(page)
    genes = main.loc[sel["votes"], ["symbol", "score", "rank", "n_detected", "n_cells", "sum_counts", "hvg_upstream"]].copy()
    genes["via_global"] = sel["global"].reindex(genes.index).fillna(False).astype(bool)
    genes["n_lineages"] = sel["lineage_genes"]["n_lineages"].reindex(genes.index).fillna(0).astype(int)
    genes["lineages"] = sel["lineage_genes"]["lineages"].reindex(genes.index).fillna("")
    genes["best_lineage_score"] = best_lin.reindex(genes.index)
    if ref is not None:
        genes["biotype"] = ref.reindex(genes.index)["biotype"]
    genes.sort_values("score", ascending=False).to_csv(os.path.join(outdir, f"{rec['key']}.genes.tsv"), sep="\t")
    return {"key": rec["key"], "species": rec["species"], "cells": meta["group_sizes"][0],
            "lineages": f"{meta.get('lineages_kept', 0)}/{meta.get('lineages_total', 0)}",
            "voted": len(sel["votes"]), "via_global": int(sel["global"].sum()), "lineage_only": len(set(sel["lineage_genes"].index) - set(main.index[sel["global"]])),
            "fallbacks": sum("failed" in " ".join(v) for v in meta.get("fit_notes", {}).values()),
            "seconds": meta.get("hvg_seconds")}


def write_index(outdir, rows, s_min, n_max):
    df = pd.DataFrame(rows).sort_values("key")
    trs = "".join(f"<tr><td><a href='{esc(r.key)}.html'>{esc(r.key)}</a></td><td>{esc(r.species)}</td><td>{r.cells:,}</td><td>{esc(r.lineages)}</td>"
                  f"<td><b>{r.voted:,}</b></td><td>{r.via_global:,}</td><td>{r.lineage_only:,}</td><td class='{'warn' if r.fallbacks else ''}'>{r.fallbacks}</td><td>{r.seconds}</td></tr>"
                  for r in df.itertuples())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>HVG reports</title><style>{CSS}</style></head><body>
<header><h1>HVG reports — {len(df)} datasets</h1><div class="sub">s_min {s_min}, n_max {n_max} · voted = genes this dataset contributes to the corpus vote</div></header>
<div class="wrap"><div class="card"><div class="kpi"><div>datasets<b>{len(df)}</b></div><div>cells<b>{int(df.cells.sum()):,}</b></div><div>median voted / dataset<b>{int(df.voted.median()):,}</b></div><div>loess fallbacks<b>{int(df.fallbacks.sum())}</b></div></div>
<table><tr><th>dataset</th><th>species</th><th>cells</th><th>lineages kept/total</th><th>voted</th><th>via global</th><th>only via lineage</th><th>loess fallbacks</th><th>sec</th></tr>{trs}</table></div></div></body></html>"""
    with open(os.path.join(outdir, "index.html"), "w") as fh:
        fh.write(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample_key", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--outdir")
    ap.add_argument("--s-min", type=float)
    ap.add_argument("--n-max", type=int)
    ap.add_argument("--min-lineage-cells", type=int)
    ap.add_argument("--min-lineages", type=int)
    a = ap.parse_args()
    cfg = featuresel.load_config(a.config)
    sel = cfg["selection"]
    s_min = a.s_min if a.s_min is not None else float(sel.get("s_min", 1.5))
    n_max = a.n_max if a.n_max is not None else int(sel.get("n_max", 3000))
    min_cells = a.min_lineage_cells if a.min_lineage_cells is not None else int(sel.get("min_lineage_cells", 200))
    min_lineages = a.min_lineages if a.min_lineages is not None else int(sel.get("min_lineages", 1))
    outdir = a.outdir or os.path.join(cfg["cache_root"], "reports")
    recs = [r for r in featuresel.scan_inputs(cfg) if os.path.exists(r["out"])]
    if not a.all:
        recs = [r for r in recs if r["key"] == a.sample_key]
        if not recs:
            raise SystemExit(f"{a.sample_key!r}: not measured yet")
    refs = {}
    rows = []
    for r in recs:
        if r["species"] not in refs:
            p = os.path.join(cfg["_dirs"]["ref"], f"biotype_{r['species']}.parquet")
            refs[r["species"]] = pd.read_parquet(p).set_index("harmonized_id") if os.path.exists(p) else None
        rows.append(build_report(cfg, r, outdir, s_min, n_max, min_cells, min_lineages, refs[r["species"]]))
        print(f"  {r['key']}: voted={rows[-1]['voted']} (global {rows[-1]['via_global']}, lineage-only {rows[-1]['lineage_only']})")
    if a.all or len(rows) > 1:
        write_index(outdir, rows, s_min, n_max)
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
