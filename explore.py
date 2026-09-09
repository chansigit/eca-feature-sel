#!/usr/bin/env python
"""Interactive explorer for one dataset's HVG call: both methods (vst, pearson),
per-group trend + score + rank curves, live thresholds. Writes a standalone HTML.

    explore.py mca3.0-brain-mouse                      # -> cache/figs/hvg_explorer_<key>.html
    explore.py mca3.0-brain-mouse --out /tmp/x.html --no-dispersion
"""
import argparse
import json
import os

import h5py
import numpy as np

import featuresel
import hvg
import worker

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTLY = os.path.join("/scratch/users/chensj16/eca-feature-sel/cache/figs", "plotly.min.js")
METHODS = ("vst", "pearson")


def measure(h5ad, rule):
    meta = {}
    with h5py.File(h5ad, "r") as f:
        g = f[f"layers/{rule.get('layer', 'counts')}"]
        enc = worker._attr(g, "encoding-type", "array")
        n_obs, n_vars = (int(x) for x in (g.attrs["shape"] if "shape" in g.attrs else g.shape))
        keep, codes, ids = worker.harmonized_map(f, n_vars)
        sym = worker._column(f, "gene_symbol_harmonized", n_vars)
        sym = np.array([str(s) if s is not None else "" for s in sym]) if sym is not None else ids
        src = hvg.Reader(g, enc, n_obs, n_vars)
        depth = hvg.cell_depth(src)
        dropped = depth < float(rule.get("min_cell_counts", 10))
        lin, names, sizes = worker.lineage_codes(f, n_obs, rule, meta, dropped)
        sizes = np.concatenate([[int((~dropped).sum())], sizes]).astype(np.int64)
        notes = {}
        out, ndet, sums, mean, var, valid = hvg.call(
            src, lin, sizes, methods=METHODS,
            span=float(rule.get("span", 0.3)), theta=float(rule.get("theta", 100)),
            min_gene_cells=int(rule.get("min_gene_cells", 3)), notes=notes)
        up = worker._column(f, "highly_variable", n_vars)
    k = keep
    return dict(ids=ids, keep=keep, sym=sym[k], names=["ALL cells"] + names, sizes=sizes,
                mean=mean[:, k], var=var[:, k], valid=valid[:, k], notes=notes,
                methods={m: {"score": out[m][:, k], "trend": out[f"{m}_trend"][:, k]} for m in METHODS},
                upstream=(np.asarray(up, dtype=bool)[k] if up is not None else None),
                n_obs=n_obs, n_dropped=int(dropped.sum()), lineages_total=meta.get("lineages_total", 0))


def seurat_dispersion(h5ad, keep, layer, n_top=2000):
    """scanpy flavor='seurat' (log1p-normalized, binned dispersion) on the whole
    dataset, for side-by-side comparison. Diagnostic only, so scanpy is fine here."""
    import anndata
    import scanpy as sc
    ad = anndata.read_h5ad(h5ad)
    ad = ad[:, keep].copy()
    ad.X = ad.layers[layer].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top, flavor="seurat")
    return {"mean": r3(ad.var["means"]), "disp_norm": r3(ad.var["dispersions_norm"]),
            "hvg": ad.var["highly_variable"].to_numpy().astype(int).tolist(), "n_top": n_top}


def r3(a):
    """Compact JSON: 4 significant digits, NaN -> null."""
    return [float(f"{x:.4g}") if np.isfinite(x) else None for x in np.asarray(a, dtype=float)]


def payload(m, key):
    n_grp = len(m["names"])
    groups = []
    for i in range(n_grp):
        ok = m["valid"][i]
        o = np.argsort(m["mean"][i][ok])
        trend_idx = np.flatnonzero(ok)[o][np.linspace(0, max(ok.sum() - 1, 0), 300).astype(int)] if ok.any() else []
        meths = {}
        for name, d in m["methods"].items():
            rank = hvg.ranks(d["score"][i:i + 1], m["valid"][i:i + 1])[0]
            meths[name] = {"score": r3(d["score"][i]), "rank": rank.tolist(),
                           "trend_x": r3(m["mean"][i][trend_idx]), "trend_y": r3(d["trend"][i][trend_idx])}
        groups.append({"name": m["names"][i], "n": int(m["sizes"][i]),
                       "mean": r3(m["mean"][i]), "var": r3(m["var"][i]),
                       "notes": m["notes"].get(i, []), "methods": meths})
    return {"key": key, "n_cells": int(m["n_obs"]), "n_dropped": m["n_dropped"],
            "n_genes": int(len(m["sym"])), "disp": m.get("disp"),
            "lineages_total": int(m["lineages_total"]),
            "symbol": m["sym"].tolist(), "id": m["ids"].tolist(),
            "upstream": (m["upstream"].astype(int).tolist() if m["upstream"] is not None else None),
            "groups": groups}


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>HVG explorer — __KEY__</title>
<script>__PLOTLY__</script>
<style>
 body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
 header{padding:14px 22px;background:#1f2937;color:#fff} header h1{margin:0;font-size:18px;font-weight:600}
 header .sub{opacity:.8;font-size:13px;margin-top:4px}
 .wrap{padding:16px 22px;max-width:1500px;margin:auto}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;margin-bottom:16px}
 .card h2{margin:0 0 8px;font-size:15px} .card p{margin:4px 0 8px;color:#444}
 .row{display:flex;gap:16px;flex-wrap:wrap} .row>.card{flex:1 1 640px;min-width:520px}
 .ctl{display:flex;gap:22px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
 .ctl label{font-size:13px;color:#333} .ctl input[type=range]{width:200px;vertical-align:middle}
 .ctl b{display:inline-block;min-width:44px;text-align:right}
 select,input[type=text]{font:inherit;padding:3px 6px}
 table{border-collapse:collapse;font-size:13px;width:100%} th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right}
 th:first-child,td:first-child{text-align:left} th{background:#f3f4f6;position:sticky;top:0}
 .hi{color:#d62728;font-weight:600} .mut{color:#888}
 .steps li{margin:4px 0} .kpi{display:flex;gap:28px;margin:6px 0 10px;flex-wrap:wrap} .kpi div{font-size:13px;color:#555} .kpi div b{display:block;font-size:22px;color:#111}
 .scroll{max-height:360px;overflow:auto} .pill{background:#1f2937;color:#fff;border-radius:4px;padding:1px 8px;font-size:12px}
</style></head><body>
<header><h1>HVG explorer — __KEY__</h1><div class="sub" id="sub"></div></header>
<div class="wrap">

<div class="card"><h2>算法（每个数据集内）</h2>
<ol class="steps">
<li><b>数值守卫</b>：总 counts &lt; 10 的细胞丢掉；在某 group 内检出 &lt; 3 个细胞的基因在该 group 不参与拟合和排名。（不是 QC，只防奇异矩阵。）</li>
<li><b>分组</b>：group 0 = 全部保留细胞；每个 coarse lineage（≥ min cells）各一个 group。</li>
<li><b>score，两种方法</b>：<span class="pill">vst</span> loess 拟合 log10(var)~log10(mean) 得期望 σ̂，score = Σ clip((x−mean)/σ̂, √N)² / (N−1)；<b>1 = 和同表达水平的典型基因一样</b>。
<span class="pill">pearson</span> μ_ij = 细胞深度 × 基因占比，NB 方差 μ+μ²/θ (θ=100)，score = Var(clip((x−μ)/√var, ±√N))；<b>1 = 纯技术噪声</b>。两者都无需归一化，都有天然参照点 1，可跨 group / 跨数据集比较。</li>
<li><b>投票</b>：基因在任一 group 里 score ≥ s_min 且 rank ≤ n_max → 该数据集投它一票（最多一票）。</li>
</ol></div>

<div class="card"><h2>阈值（实时）</h2>
<div class="ctl">
 <label>方法 <select id="method"><option value="vst">vst (seurat_v3)</option><option value="pearson">pearson residuals</option></select></label>
 <label>s_min（score 下限）<input type="range" id="smin" min="1" max="10" step="0.05" value="1.5"> <b id="sminv">1.50</b></label>
 <label>n_max（每 group 排名上限）<input type="range" id="nmax" min="100" max="6000" step="100" value="3000"> <b id="nmaxv">3000</b></label>
 <label>lineage 最少细胞<input type="range" id="mincells" min="200" max="5000" step="100" value="200"> <b id="mincellsv">200</b></label>
 <label><input type="checkbox" id="fixed"> 对照：固定 top-2000（忽略 s_min）</label>
</div>
<div class="kpi">
 <div>该数据集投票的基因<b id="k_union">–</b></div>
 <div>仅靠 group 0 (全局)<b id="k_global">–</b></div>
 <div>仅靠 lineage 补进来的<b id="k_lin_only">–</b></div>
 <div>参与的 lineage<b id="k_groups">–</b></div>
 <div>另一方法同阈值选中 / 两者交集<b id="k_other">–</b></div>
 <div>与 rsi 上游 2000 HVG 的重叠<b id="k_up">–</b></div>
</div>
<div class="scroll"><table id="gtab"></table></div>
</div>

<div class="row">
 <div class="card"><h2 id="tA">A. 均值–方差 与 期望方差曲线</h2>
  <div class="ctl"><label>group <select id="gsel"></select></label><span class="mut" id="gnote"></span></div>
  <div id="pA" style="height:460px"></div></div>
 <div class="card"><h2>B. score 随表达水平（标准化后应当是平的）</h2>
  <p class="mut">红 = 在当前阈值下被该 group 选中；虚线 = score 1 和 s_min。</p>
  <div id="pB" style="height:460px"></div></div>
</div>
<div class="row">
 <div class="card"><h2>C. 每个 group 的 rank–score 曲线</h2>
  <p class="mut">竖虚线 = n_max，横虚线 = s_min。</p>
  <div id="pC" style="height:460px"></div></div>
 <div class="card"><h2>D. 每个 group 过线的基因数</h2>
  <div id="pD" style="height:460px"></div></div>
</div>

<div class="card"><h2>vst vs pearson（group 0，全部基因）</h2>
<p class="mut">每个点一个基因；红 = 当前阈值下两种方法都选中，蓝 = 只有 vst，橙 = 只有 pearson。对角线 = 两者一致。</p>
<div id="pF" style="height:480px"></div></div>

<div class="card" id="dispcard"><h2>对照：老方法 <code>flavor='seurat'</code>（log1p 归一化 → dispersion → 按 mean 分 20 bin 做 z-score）</h2>
<p class="mut">rsi 上游用的就是这个。x 轴是 log1p 归一化均值，y 是 bin 内 z-score。左：老方法自己的 top-2000；右：<b>当前方法+滑块</b>选中的基因落在老方法坐标里的位置。</p>
<div class="kpi">
 <div>老方法 top-2000 ∩ 当前选中<b id="k_ov">–</b></div>
 <div>只有当前方法选中<b id="k_vst_only">–</b></div>
 <div>只有老方法选中<b id="k_disp_only">–</b></div>
</div>
<div class="row" style="gap:12px">
 <div style="flex:1 1 480px"><div id="pE1" style="height:420px"></div></div>
 <div style="flex:1 1 480px"><div id="pE2" style="height:420px"></div></div>
</div></div>

<div class="card"><h2>查基因</h2>
<div class="ctl"><label>symbol / Ensembl id <input type="text" id="q" placeholder="Gapdh"></label></div>
<div class="scroll"><table id="qtab"></table></div></div>

</div>
<script>
const D = __DATA__;
const G = D.groups, NG = G.length, NGENE = D.n_genes;
const el = id => document.getElementById(id);
el('sub').textContent = `${D.n_cells.toLocaleString()} cells (${D.n_dropped} dropped by the counts guard) · ${NGENE.toLocaleString()} harmonized genes · lineages kept ${NG-1}/${D.lineages_total} · ` + G.map(g=>`${g.name} (${g.n})`).join(' · ');
G.forEach((g,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${g.name}  (N=${g.n})`;el('gsel').appendChild(o);});

const M = () => el('method').value;
const other = m => m==='vst' ? 'pearson' : 'vst';
const S = (i,m) => G[i].methods[m||M()];
function params(){return {m:M(), smin:+el('smin').value, nmax:+el('nmax').value, mincells:+el('mincells').value, fixed:el('fixed').checked};}
function active(p){return G.map((g,i)=> i===0 || g.n>=p.mincells);}
function passes(i,m,j,p){const s=S(i,m); const r=s.rank[j]; if(!r) return false; if(p.fixed) return r<=2000; return s.score[j]!=null && s.score[j]>=p.smin && r<=p.nmax;}
function hits(m,p,act){const hg=new Uint8Array(NGENE), hl=new Uint8Array(NGENE), per=G.map(()=>0);
  for(let j=0;j<NGENE;j++) for(let i=0;i<NG;i++){ if(!act[i]) continue; if(passes(i,m,j,p)){ per[i]++; if(i===0) hg[j]=1; else hl[j]=1; } }
  return {hg,hl,per};}

function recompute(){
  const p=params(); el('sminv').textContent=p.smin.toFixed(2); el('nmaxv').textContent=p.nmax; el('mincellsv').textContent=p.mincells;
  const act=active(p); const H=hits(p.m,p,act), O=hits(other(p.m),p,act);
  let u=0,g0=0,lo=0,up=0,upn=0,ou=0,both=0;
  for(let j=0;j<NGENE;j++){const h=H.hg[j]||H.hl[j], o=O.hg[j]||O.hl[j]; if(h){u++; if(D.upstream&&D.upstream[j])up++;} if(H.hg[j])g0++; if(H.hl[j]&&!H.hg[j])lo++; if(D.upstream&&D.upstream[j])upn++; if(o)ou++; if(h&&o)both++;}
  el('k_union').textContent=u.toLocaleString(); el('k_global').textContent=g0.toLocaleString(); el('k_lin_only').textContent=lo.toLocaleString();
  el('k_groups').textContent=`${act.filter((a,i)=>a&&i>0).length} / ${NG-1}`; el('k_up').textContent=D.upstream?`${up} / ${upn}`:'n/a';
  el('k_other').textContent=`${ou.toLocaleString()} / ${both.toLocaleString()}`;
  let h=`<tr><th>group</th><th>N cells</th><th>过线基因 (${p.m})</th><th>过线基因 (${other(p.m)})</th><th>score@rank 2000</th><th>#score≥1.5</th><th>#score≥2</th><th>#score≥3</th><th>vst 拟合</th></tr>`;
  G.forEach((g,i)=>{ const s=S(i,p.m); let s2000=null,c15=0,c2=0,c3=0; for(let j=0;j<NGENE;j++){const v=s.score[j]; if(v==null||!s.rank[j])continue; if(s.rank[j]===2000)s2000=v; if(v>=1.5)c15++; if(v>=2)c2++; if(v>=3)c3++;}
    h+=`<tr class="${act[i]?'':'mut'}"><td>${g.name}${act[i]?'':' (skipped: too small)'}</td><td>${g.n}</td><td class="hi">${act[i]?H.per[i]:'–'}</td><td>${act[i]?O.per[i]:'–'}</td><td>${s2000==null?'–':s2000.toFixed(2)}</td><td>${c15}</td><td>${c2}</td><td>${c3}</td><td>${g.notes.length?g.notes.join('; '):'loess'}</td></tr>`;});
  el('gtab').innerHTML=h;
  drawAB(p); drawC(p,act); drawD(H.per,act,p); drawF(p,H,O); drawE(H);
}

function drawAB(p){
  const i=+el('gsel').value, g=G[i], s=S(i,p.m); el('gnote').textContent=g.notes.length?('vst 拟合: '+g.notes.join('; ')):'';
  el('tA').textContent = p.m==='vst' ? 'A. 均值–方差 与 loess 趋势（vst）' : 'A. 均值–方差 与 NB 期望方差 μ+μ²/θ（pearson，取平均深度）';
  const sel=[],bg=[]; for(let j=0;j<NGENE;j++){ if(g.mean[j]==null||g.mean[j]<=0||g.var[j]==null||g.var[j]<=0) continue; (passes(i,p.m,j,p)?sel:bg).push(j); }
  const lx=a=>a.map(j=>Math.log10(g.mean[j])), ly=a=>a.map(j=>Math.log10(g.var[j]));
  const tx=a=>a.map(j=>`${D.symbol[j]}<br>mean ${g.mean[j]}<br>vst ${S(i,'vst').score[j]} (rank ${S(i,'vst').rank[j]})<br>pearson ${S(i,'pearson').score[j]} (rank ${S(i,'pearson').rank[j]})`);
  Plotly.react('pA',[
    {type:'scattergl',mode:'markers',x:lx(bg),y:ly(bg),text:tx(bg),hoverinfo:'text',marker:{size:3,color:'#bbb'},name:'other genes'},
    {type:'scattergl',mode:'markers',x:lx(sel),y:ly(sel),text:tx(sel),hoverinfo:'text',marker:{size:4,color:'#d62728'},name:`selected in ${g.name} (${sel.length})`},
    {type:'scatter',mode:'lines',x:s.trend_x.map(Math.log10),y:s.trend_y.map(Math.log10),line:{color:'#111',width:2},name:p.m==='vst'?'loess trend (expected var)':'NB expected var'}],
    {margin:{t:10,l:55,r:10,b:45},xaxis:{title:'log10 mean (raw counts)'},yaxis:{title:'log10 variance'},legend:{x:0.01,y:0.99}},{responsive:true});
  const sy=a=>a.map(j=>s.score[j]);
  Plotly.react('pB',[
    {type:'scattergl',mode:'markers',x:lx(bg),y:sy(bg),text:tx(bg),hoverinfo:'text',marker:{size:3,color:'#bbb'},name:'other genes'},
    {type:'scattergl',mode:'markers',x:lx(sel),y:sy(sel),text:tx(sel),hoverinfo:'text',marker:{size:4,color:'#d62728'},name:'selected'}],
    {margin:{t:10,l:55,r:10,b:45},xaxis:{title:'log10 mean'},yaxis:{title:`score (${p.m})`,type:'log'},legend:{x:0.01,y:0.99},
     shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{dash:'dash',color:'#111',width:1}},
             {type:'line',xref:'paper',x0:0,x1:1,y0:p.smin,y1:p.smin,line:{dash:'dash',color:'#1f77b4',width:1}}]},{responsive:true});
}
function drawC(p,act){
  const tr=G.map((g,i)=>{const s=S(i,p.m); const v=s.score.filter((x,j)=>x!=null&&x>0&&s.rank[j]>0).sort((a,b)=>b-a); return {type:'scattergl',mode:'lines',x:v.map((_,k)=>k+1),y:v,name:`${g.name} (N=${g.n})`,line:{width:act[i]?1.6:0.8,dash:act[i]?'solid':'dot'}};});
  Plotly.react('pC',tr,{margin:{t:10,l:55,r:10,b:45},xaxis:{title:'rank within group',type:'log'},yaxis:{title:`score (${p.m})`,type:'log'},legend:{font:{size:11}},
    shapes:[{type:'line',yref:'paper',x0:p.fixed?2000:p.nmax,x1:p.fixed?2000:p.nmax,y0:0,y1:1,line:{dash:'dot',color:'#111',width:1}},
            {type:'line',xref:'paper',x0:0,x1:1,y0:p.fixed?1:p.smin,y1:p.fixed?1:p.smin,line:{dash:'dash',color:'#1f77b4',width:1}}]},{responsive:true});
}
function drawD(per,act,p){
  Plotly.react('pD',[{type:'bar',x:G.map(g=>`${g.name}<br>N=${g.n}`),y:per.map((v,i)=>act[i]?v:0),marker:{color:G.map((g,i)=>i===0?'#1f77b4':'#ff7f0e')}}],
    {margin:{t:10,l:55,r:10,b:120},yaxis:{title:`# genes passing (${p.m})`},xaxis:{tickfont:{size:10}}},{responsive:true});
}
function drawF(p,H,O){
  const v=S(0,'vst'), q=S(0,'pearson'); const both=[],onlyV=[],onlyP=[],none=[];
  const hv = p.m==='vst'?H:O, hp = p.m==='pearson'?H:O;
  for(let j=0;j<NGENE;j++){ if(v.score[j]==null||q.score[j]==null||!v.rank[j]||!q.rank[j]) continue; const a=hv.hg[j], b=hp.hg[j]; (a&&b?both:a?onlyV:b?onlyP:none).push(j); }
  const X=a=>a.map(j=>v.score[j]), Y=a=>a.map(j=>q.score[j]), T=a=>a.map(j=>`${D.symbol[j]}<br>vst ${v.score[j]} (rank ${v.rank[j]})<br>pearson ${q.score[j]} (rank ${q.rank[j]})`);
  const mk=(a,c,n,sz)=>({type:'scattergl',mode:'markers',x:X(a),y:Y(a),text:T(a),hoverinfo:'text',marker:{size:sz,color:c},name:`${n} (${a.length})`});
  Plotly.react('pF',[mk(none,'#ccc','neither',3),mk(onlyV,'#1f77b4','vst only',4),mk(onlyP,'#ff7f0e','pearson only',4),mk(both,'#d62728','both',4),
    {type:'scatter',mode:'lines',x:[0.3,300],y:[0.3,300],line:{color:'#888',dash:'dot',width:1},name:'y = x',hoverinfo:'skip'}],
    {margin:{t:10,l:60,r:10,b:50},xaxis:{title:'vst score (group 0)',type:'log'},yaxis:{title:'pearson score (group 0)',type:'log'},legend:{x:0.01,y:0.99}},{responsive:true});
}
function drawE(H){
  const E=D.disp; if(!E){el('dispcard').style.display='none';return;}
  const p=params(); const bgV=[],selV=[],bgO=[],selO=[]; let ov=0,vo=0,dO=0;
  for(let j=0;j<NGENE;j++){ if(E.mean[j]==null||E.disp_norm[j]==null) continue;
    const v=H.hg[j]||H.hl[j], o=E.hvg[j]; (o?selO:bgO).push(j); (v?selV:bgV).push(j);
    if(v&&o)ov++; else if(v)vo++; else if(o)dO++; }
  el('k_ov').textContent=ov; el('k_vst_only').textContent=vo; el('k_disp_only').textContent=dO;
  const X=a=>a.map(j=>E.mean[j]), Y=a=>a.map(j=>E.disp_norm[j]);
  const T=a=>a.map(j=>`${D.symbol[j]}<br>${p.m} ${S(0,p.m).score[j]} (rank ${S(0,p.m).rank[j]})<br>disp_norm ${E.disp_norm[j]}`);
  const lay=t=>({margin:{t:28,l:55,r:10,b:45},title:{text:t,font:{size:13}},xaxis:{title:'log1p 归一化均值'},yaxis:{title:'dispersions_norm (bin 内 z-score)'},legend:{x:0.55,y:0.99}});
  Plotly.react('pE1',[{type:'scattergl',mode:'markers',x:X(bgO),y:Y(bgO),text:T(bgO),hoverinfo:'text',marker:{size:3,color:'#bbb'},name:'other'},
    {type:'scattergl',mode:'markers',x:X(selO),y:Y(selO),text:T(selO),hoverinfo:'text',marker:{size:4,color:'#d62728'},name:`seurat flavor top-${E.n_top}`}],lay('老方法自己选的'),{responsive:true});
  Plotly.react('pE2',[{type:'scattergl',mode:'markers',x:X(bgV),y:Y(bgV),text:T(bgV),hoverinfo:'text',marker:{size:3,color:'#bbb'},name:'other'},
    {type:'scattergl',mode:'markers',x:X(selV),y:Y(selV),text:T(selV),hoverinfo:'text',marker:{size:4,color:'#2ca02c'},name:`${p.m} 当前阈值选中 (${selV.length})`}],lay(`${p.m}（当前滑块）选的，画在老方法的坐标里`),{responsive:true});
}
function lookup(){
  const q=el('q').value.trim().toLowerCase(); if(!q){el('qtab').innerHTML='';return;}
  const idx=[]; for(let j=0;j<NGENE&&idx.length<25;j++){ if(D.symbol[j].toLowerCase()===q||D.id[j].toLowerCase()===q) idx.unshift(j); else if(D.symbol[j].toLowerCase().startsWith(q)) idx.push(j); }
  const p=params(); let h='<tr><th>gene</th>'+G.map(g=>`<th>${g.name}<br><span class="mut">vst s/r · pearson s/r</span></th>`).join('')+'<th>rsi 上游</th></tr>';
  const f=(i,m,j)=>{const s=S(i,m); return `<span class="${passes(i,m,j,p)?'hi':''}">${s.score[j]==null?'–':s.score[j].toFixed(2)}/${s.rank[j]||'–'}</span>`;};
  idx.forEach(j=>{ h+=`<tr><td><b>${D.symbol[j]}</b> <span class="mut">${D.id[j]}</span></td>`+G.map((g,i)=>`<td>${f(i,'vst',j)} · ${f(i,'pearson',j)}</td>`).join('')+`<td>${D.upstream?(D.upstream[j]?'yes':'no'):'n/a'}</td></tr>`;});
  el('qtab').innerHTML=h;
}
['method','smin','nmax','mincells','fixed'].forEach(id=>el(id).addEventListener('input',()=>{recompute();lookup();}));
el('gsel').addEventListener('change',()=>drawAB(params())); el('q').addEventListener('input',lookup);
recompute();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample_key")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--out")
    ap.add_argument("--no-dispersion", action="store_true",
                    help="skip the legacy seurat-flavor comparison (it loads the AnnData)")
    a = ap.parse_args()
    cfg = featuresel.load_config(a.config)
    rec = next((r for r in featuresel.scan_inputs(cfg) if r["key"] == a.sample_key), None)
    if rec is None:
        raise SystemExit(f"{a.sample_key!r} not in inputs")
    m = measure(rec["h5ad"], cfg["hvg"])
    if not a.no_dispersion:
        m["disp"] = seurat_dispersion(rec["h5ad"], m["keep"], cfg["hvg"].get("layer", "counts"))
    data = json.dumps(payload(m, a.sample_key), separators=(",", ":"))
    plotly = open(PLOTLY).read() if os.path.exists(PLOTLY) else \
        'document.write(\'<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\\/script>\')'
    html = HTML.replace("__KEY__", a.sample_key).replace("__PLOTLY__", plotly).replace("__DATA__", data)
    out = a.out or os.path.join(cfg["cache_root"], "figs", f"hvg_explorer_{a.sample_key}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"{out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
