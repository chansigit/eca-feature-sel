"""Corpus vote explorer: per-category counts of genes passing the cross-dataset
vote, with s_min and the vote cutoff adjustable in the page.

    python votes.py            -> cache/figs/votes_<species>.html

One dataset = one vote (dataset-level group OR >= min_lineages lineages).
Votes are precomputed on a grid of s_min values; n_max / min_lineage_cells /
min_lineages come from config.yaml.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import featuresel as fs
from explore import PLOTLY

S_GRID = [round(1.0 + 0.1 * i, 1) for i in range(21)]  # 1.0 .. 3.0
FLAGS = [c for c in fs.FLAGCOLS if c != "is_protein_coding"]


def pair_best(main, lin, n_max, min_cells, min_lineages):
    """Per (dataset, gene): best score through which the dataset could vote."""
    ok = (main["rank"] > 0) & (main["rank"] <= n_max)
    g = main.loc[ok, ["sample_key", "harmonized_id", "score"]]
    lk = lin[(lin["lineage_cells"] >= min_cells) & (lin["rank"] > 0) & (lin["rank"] <= n_max)]
    lk = lk.sort_values("score", ascending=False)
    nth = lk.groupby(["sample_key", "harmonized_id"]).cumcount() == min_lineages - 1
    l = lk.loc[nth, ["sample_key", "harmonized_id", "score"]]  # k-th best lineage score
    both = pd.concat([g, l], ignore_index=True)
    return both.groupby(["sample_key", "harmonized_id"])["score"].max()


def payload(main, lin, ref, cfg):
    sel = cfg["selection"]
    n_max = int(sel.get("n_max", 3000))
    pb = pair_best(main, lin, n_max, sel.get("min_lineage_cells", 200), sel.get("min_lineages", 1))
    gene = pb.index.get_level_values("harmonized_id")
    votes = pd.DataFrame({f"{s}": (pb.values >= s) for s in S_GRID}, index=gene).groupby(level=0).sum()
    present = main.groupby("harmonized_id")["sample_key"].nunique()
    sym = main.dropna(subset=["symbol"]).drop_duplicates("harmonized_id").set_index("harmonized_id")["symbol"]
    t = pd.DataFrame(index=present.index)
    t["present"] = present
    t = t.join(votes, how="left").fillna(0)
    t = t.join(ref.set_index("harmonized_id")[["biotype"] + FLAGS], how="left")
    t["biotype"] = t["biotype"].fillna("unknown")
    t["symbol"] = sym.reindex(t.index).fillna(t.index.to_series())
    bio = sorted(t["biotype"].unique(), key=lambda b: -(t["biotype"] == b).sum())
    flagbits = np.zeros(len(t), dtype=np.int64)
    for i, f in enumerate(FLAGS):
        flagbits |= (t[f].fillna(False).astype(bool).values.astype(np.int64) << i)
    return {
        "s_grid": S_GRID, "n_max": n_max, "n_datasets": int(main["sample_key"].nunique()),
        "biotypes": bio, "flags": FLAGS,
        "rules": sel.get("hvg_min_datasets", {}), "drop": sel.get("category_drop", []),
        "keep": sel.get("category_keep", []),
        "id": t.index.tolist(), "symbol": t["symbol"].tolist(),
        "bio": t["biotype"].map({b: i for i, b in enumerate(bio)}).astype(int).tolist(),
        "flag": flagbits.tolist(), "present": t["present"].astype(int).tolist(),
        "votes": t[[f"{s}" for s in S_GRID]].astype(int).values.T.tolist(),
    }


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>vote explorer — __SP__</title>
<script>__PLOTLY__</script>
<style>
 body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
 header{padding:14px 22px;background:#1f2937;color:#fff} header h1{margin:0;font-size:18px;font-weight:600}
 header .sub{opacity:.8;font-size:13px;margin-top:4px}
 .wrap{padding:16px 22px;max-width:1500px;margin:auto}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;margin-bottom:16px}
 .card h2{margin:0 0 8px;font-size:15px} .card p{margin:4px 0 8px;color:#444}
 .row{display:flex;gap:16px;flex-wrap:wrap} .row>.card{flex:1 1 560px;min-width:480px}
 .ctl{display:flex;gap:22px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
 .ctl label{font-size:13px;color:#333} .ctl input[type=range]{width:220px;vertical-align:middle}
 .ctl b{display:inline-block;min-width:44px;text-align:right}
 table{border-collapse:collapse;font-size:13px;width:100%} th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right}
 th:first-child,td:first-child{text-align:left} th{background:#f3f4f6;position:sticky;top:0}
 tr.sel td{background:#fff7d6} tbody tr{cursor:pointer} tbody tr:hover td{background:#f6f6f6}
 .kpi{display:flex;gap:28px;margin:6px 0 10px;flex-wrap:wrap} .kpi div{font-size:13px;color:#555} .kpi div b{display:block;font-size:22px;color:#111}
 .scroll{max-height:420px;overflow:auto} textarea{width:100%;height:160px;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace}
 .mut{color:#888}
</style></head><body>
<header><h1>Corpus vote explorer — __SP__</h1><div class="sub" id="sub"></div></header>
<div class="wrap">

<div class="card"><h2>阈值（实时）</h2>
<div class="ctl">
 <label>s_min <input type="range" id="smin" min="0" max="0" step="1"> <b id="sminv"></b></label>
 <label>投票 cutoff（≥ 多少个数据集） <input type="range" id="cut" min="1" max="60" step="1" value="3"> <b id="cutv"></b></label>
 <span class="mut">n_max = <span id="nmax"></span>（固定，来自 config）· 一数据集一票（全局 或 lineage）</span>
</div>
<div class="kpi">
 <div><b id="k_all"></b>全部类别通过</div>
 <div><b id="k_pc"></b>protein_coding 通过</div>
 <div><b id="k_lnc"></b>lncRNA 通过</div>
 <div><b id="k_cfg"></b>按 config 规则通过 <span class="mut" id="cfgrule"></span></div>
</div>
</div>

<div class="row">
<div class="card"><h2>按 biotype</h2><p class="mut">点击一行查看它的票数分布和基因列表。“出现” = 在 ≥1 个数据集里被测到。</p>
<div class="scroll"><table id="tb_bio"><thead><tr><th>biotype</th><th>出现</th><th>≥1 票</th><th>通过</th><th>通过%</th></tr></thead><tbody></tbody></table></div></div>
<div class="card"><h2>按 flag（与 biotype 有重叠）</h2><p class="mut">config 黑名单：<span id="drop"></span>；白名单：<span id="keep"></span></p>
<div class="scroll"><table id="tb_flag"><thead><tr><th>flag</th><th>出现</th><th>≥1 票</th><th>通过</th><th>通过%</th></tr></thead><tbody></tbody></table></div></div>
</div>

<div class="row">
<div class="card"><h2 id="h_hist">票数分布</h2><div id="hist" style="height:340px"></div></div>
<div class="card"><h2 id="h_genes">基因</h2><p class="mut">通过的基因，按票数降序（symbol:票数）</p><textarea id="genes" readonly></textarea>
<p class="mut" style="margin-top:8px">未通过但票数最高的 60 个：</p><textarea id="near" readonly style="height:90px"></textarea></div>
</div>

</div>
<script>
const D = __DATA__;
const N = D.id.length;
const $ = id => document.getElementById(id);
let cat = {kind:'bio', idx:0};
$('nmax').textContent = D.n_max; $('drop').textContent = D.drop.join(', ')||'无'; $('keep').textContent = D.keep.join(', ')||'无';
$('sub').textContent = `${D.n_datasets} 个数据集 · ${N.toLocaleString()} 个基因被测到 · s_min 网格 ${D.s_grid[0]}–${D.s_grid[D.s_grid.length-1]}`;
const smin = $('smin'); smin.max = D.s_grid.length-1; smin.value = D.s_grid.indexOf(1.5) >= 0 ? D.s_grid.indexOf(1.5) : 0;
function members(kind, idx){ // boolean mask for a category
  const m = new Uint8Array(N);
  if (kind==='bio') for (let i=0;i<N;i++) m[i] = D.bio[i]===idx;
  else for (let i=0;i<N;i++) m[i] = (D.flag[i]>>idx)&1;
  return m;
}
function ruleFor(i){ // config hvg_min_datasets: first-match-wins on flags/biotype, else default
  const r = D.rules; if (typeof r === 'number') return r;
  for (const k in r){ if (k==='default') continue;
    if (D.biotypes[D.bio[i]]===k) return r[k];
    const fi = D.flags.indexOf(k); if (k==='is_protein_coding' ? D.biotypes[D.bio[i]]==='protein_coding' : (fi>=0 && ((D.flag[i]>>fi)&1))) return r[k]; }
  return r.default ?? 1;
}
function inList(i, list){ return list.some(k => D.biotypes[D.bio[i]]===k || (D.flags.indexOf(k)>=0 && ((D.flag[i]>>D.flags.indexOf(k))&1)) || (k==='is_protein_coding' && D.biotypes[D.bio[i]]==='protein_coding')); }
function fill(tbId, kind, names){
  const v = D.votes[+smin.value], cut = +$('cut').value;
  const rows = names.map((nm, idx) => { const m = members(kind, idx); let seen=0, any=0, pass=0;
    for (let i=0;i<N;i++) if (m[i]) { seen++; if (v[i]>=1) any++; if (v[i]>=cut) pass++; }
    return {nm, idx, seen, any, pass}; }).filter(r => r.seen>0).sort((a,b)=>b.pass-a.pass||b.seen-a.seen);
  const tb = $(tbId).querySelector('tbody'); tb.innerHTML = '';
  for (const r of rows){ const tr = document.createElement('tr');
    tr.className = (cat.kind===kind && cat.idx===r.idx) ? 'sel' : '';
    tr.innerHTML = `<td>${r.nm}</td><td>${r.seen.toLocaleString()}</td><td>${r.any.toLocaleString()}</td><td><b>${r.pass.toLocaleString()}</b></td><td>${(100*r.pass/r.seen).toFixed(1)}</td>`;
    tr.onclick = () => { cat = {kind, idx:r.idx}; update(); }; tb.appendChild(tr); }
}
function update(){
  const si = +smin.value, s = D.s_grid[si], v = D.votes[si], cut = +$('cut').value;
  $('sminv').textContent = s.toFixed(1); $('cutv').textContent = cut;
  let all=0, pc=0, lnc=0, cfg=0;
  for (let i=0;i<N;i++){ const b = D.biotypes[D.bio[i]];
    if (v[i]>=cut){ all++; if (b==='protein_coding') pc++; if (b==='lncRNA') lnc++; }
    if (v[i]>=ruleFor(i) && (inList(i,D.keep) || !inList(i,D.drop))) cfg++; }
  $('k_all').textContent = all.toLocaleString(); $('k_pc').textContent = pc.toLocaleString(); $('k_lnc').textContent = lnc.toLocaleString();
  $('k_cfg').textContent = cfg.toLocaleString(); $('cfgrule').textContent = JSON.stringify(D.rules);
  fill('tb_bio','bio',D.biotypes); fill('tb_flag','flag',D.flags);
  // histogram + gene list for selected category
  const m = members(cat.kind, cat.idx), name = cat.kind==='bio' ? D.biotypes[cat.idx] : D.flags[cat.idx];
  const h = new Array(D.n_datasets+1).fill(0), passed = [], near = [];
  for (let i=0;i<N;i++) if (m[i]) { h[v[i]]++; if (v[i]>=cut) passed.push(i); else if (v[i]>0) near.push(i); }
  passed.sort((a,b)=>v[b]-v[a]); near.sort((a,b)=>v[b]-v[a]);
  $('h_hist').textContent = `票数分布 — ${name}（s_min ${s.toFixed(1)}）`;
  $('h_genes').textContent = `${name}：通过 ${passed.length.toLocaleString()} 个`;
  $('genes').value = passed.map(i => `${D.symbol[i]}:${v[i]}`).join('  ');
  $('near').value = near.slice(0,60).map(i => `${D.symbol[i]}:${v[i]}`).join('  ');
  const x = [], y = [], c = []; for (let k=0;k<=D.n_datasets;k++) if (h[k]) { x.push(k); y.push(h[k]); c.push(k>=cut ? '#d62728' : '#9ca3af'); }
  Plotly.react('hist', [{type:'bar', x, y, marker:{color:c}, hovertemplate:'%{x} 票: %{y} 基因<extra></extra>'}],
    {margin:{l:50,r:10,t:10,b:40}, xaxis:{title:'票数（数据集数）'}, yaxis:{title:'基因数', type:'log'},
     shapes:[{type:'line', x0:cut-0.5, x1:cut-0.5, y0:0, y1:1, yref:'paper', line:{color:'#d62728', dash:'dot'}}]}, {displaylogo:false, responsive:true});
}
smin.oninput = update; $('cut').oninput = update; update();
</script></body></html>
"""


def main():
    cfg = fs.load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    main_df, lin = fs._load_stats(cfg, fs.scan_inputs(cfg))
    plotly = open(PLOTLY, encoding="utf-8").read()
    for sp in sorted(main_df["species"].unique()):
        ref = pd.read_parquet(os.path.join(cfg["_dirs"]["ref"], f"biotype_{sp}.parquet"))
        sub = main_df[main_df["species"] == sp]
        p = payload(sub, lin[lin["species"] == sp], ref, cfg)
        out = os.path.join(cfg["cache_root"], "figs", f"votes_{sp}.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        html = (HTML.replace("__SP__", sp).replace("__PLOTLY__", plotly)
                .replace("__DATA__", json.dumps(p, separators=(",", ":"))))
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(html)
        v15 = p["votes"][S_GRID.index(1.5)]
        print(f"{sp}: {len(p['id'])} genes, {sum(1 for x in v15 if x >= 3)} pass >=3 at s_min 1.5 -> {out}")


if __name__ == "__main__":
    main()
