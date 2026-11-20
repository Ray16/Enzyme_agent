"""Interactive 3D viewer (single self-contained HTML, 3Dmol.js) for substrate_channels results.

One dropdown entry per enzyme; each entry shows up to two linked viewers side by side (e.g. ATP*Mg kept
vs bare pocket) with the FULL enzyme as a light cartoon, ligands as sticks (gray carbons; Mg = green
spheres) and the CAVER tunnels as radius-coloured translucent spheres (same colormap as render_channels:
narrow = red, wide = blue). A table under each viewer lists every reported channel. No ray tracing, so
building the page takes seconds; rotation answers "is X inside the tunnel?" that a 2D image cannot.

  python -m structural_modeling_agent.analysis.channels_html --out channels.html \\
      --entry "GlnA3 (Q9KZC7)" "ATP·Mg kept=RUN_A" "bare pocket=RUN_B" \\
      --entry "BacD (P39641)"  "ATP·Mg kept=RUN_C" ...

Frame: the protein + ligands are taken from the run's first provenance pose (the superposition
reference), which is the frame CAVER's tunnel spheres are in. Descriptive only -- no verdict.
Needs a browser with internet access (3Dmol.js is loaded from its CDN).
"""
import argparse
import json
import os
import sys
import tempfile
import warnings

from . import render_channels as rc
from . import substrate_channels as sch


def _pdb_text(path, keep):
    """ATOM/HETATM lines of `path` (PDB) passing keep(line); hydrogens dropped to keep the page small."""
    out = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")) and keep(ln):
            el = ln[76:78].strip() or ln[12:16].strip()[:1]
            if el.upper() != "H":
                out.append(ln.rstrip("\n"))
    return "\n".join(out) + "\nEND\n"


def _tunnels(run_dir, channels):
    data = os.path.join(run_dir, "caver_out", "data")
    out = []
    for ch in channels:
        tp = rc._tunnel_pdb(data, ch["cluster"])
        if not tp:
            continue
        sph = []
        for ln in open(tp):
            if ln.startswith(("ATOM", "HETATM")):
                try:
                    sph.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54]), float(ln[60:66])])
                except ValueError:
                    pass
        cols = ["#%02x%02x%02x" % tuple(int(255 * v) for v in c) for c in rc._sphere_colors([s[3] for s in sph])]
        out.append({"cluster": ch["cluster"], "spheres": [s + [c] for s, c in zip(sph, cols)]})
    return out


def panel(run_dir, label):
    res = json.load(open(os.path.join(run_dir, "channels.json")))
    channels = res.get("channels", [])
    poses = res.get("provenance", {}).get("poses", [])
    holo = next((p for p in poses if os.path.exists(p)), None)
    if holo is None:
        raise FileNotFoundError(f"no provenance pose found for {run_dir}")
    tmp = tempfile.mkdtemp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = _pdb_text(sch._to_pdb(holo, tmp), lambda ln: ln.startswith("ATOM"))
        ligs = {k: _pdb_text(p, lambda ln: True) for k, p in rc._split_ligand_pdbs(holo, tmp).items()}
    rows = [{"cluster": c["cluster"], "freq": c["frequency"], "bottleneck": c["bottleneck_A"]["mean"],
             "bottleneck_sd": c["bottleneck_A"]["sd"], "length": c["length_A"]["mean"],
             "stable": (c.get("stability") or {}).get("fraction")} for c in channels]
    return {"label": label, "run_dir": os.path.abspath(run_dir), "n_poses": res.get("n_poses"),
            "obstacles": res.get("obstacles"), "notes": res.get("notes", []),
            "protein": prot, "ligands": ligs, "tunnels": _tunnels(run_dir, channels), "channels": rows}


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
<style>
 body{font-family:Helvetica,Arial,sans-serif;margin:12px;color:#222}
 h1{font-size:20px;margin:4px 0 8px} select{font-size:15px;padding:3px}
 .row{display:flex;gap:12px;margin-top:10px} .col{flex:1;min-width:0}
 .vw{width:100%;height:620px;position:relative;border:1px solid #bbb;border-radius:6px}
 h2{font-size:15px;margin:4px 0} table{border-collapse:collapse;font-size:13px;margin-top:6px}
 td,th{border:1px solid #ccc;padding:2px 7px;text-align:right} th{background:#f2f2f2}
 .note{font-size:12px;color:#555;margin-top:4px} .bar{display:inline-block;width:220px;height:12px;
 background:linear-gradient(to right,__GRAD__);vertical-align:middle;border:1px solid #999}
 label{font-size:13px;margin-left:14px}
</style></head><body>
<h1>__TITLE__</h1>
<div>Enzyme: <select id="sel"></select>
 <label><input type="checkbox" id="cart" checked> cartoon</label>
 <label><input type="checkbox" id="surf"> protein surface</label>
 <label><input type="checkbox" id="sync" checked> link views</label>
 <span style="margin-left:18px;font-size:13px">tunnel radius __VMIN__ Å <span class="bar"></span> __VMAX__ Å</span></div>
<div class="note">Gray cartoon = full enzyme · gray sticks = substrate (ball-and-stick) and ATP (thin) · green = Mg ·
 coloured spheres = CAVER tunnels (red = narrow). Drag to rotate, scroll to zoom, hover a tunnel for its channel id.</div>
<div class="row" id="row"></div>
<script>
const DATA = __DATA__;
const sel = document.getElementById('sel');
DATA.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=e.name;sel.appendChild(o);});
let viewers=[];
function fmt(x,d){return (x===null||x===undefined)?'n/a':Number(x).toFixed(d);}
function draw(){
  const e=DATA[sel.value], row=document.getElementById('row'); row.innerHTML=''; viewers=[];
  e.panels.forEach((p,k)=>{
    const col=document.createElement('div'); col.className='col';
    col.innerHTML=`<h2>${p.label} <span style="font-weight:normal;color:#666">(${p.n_poses} poses; obstacles: ${p.obstacles})</span></h2>`;
    const vd=document.createElement('div'); vd.className='vw'; col.appendChild(vd);
    let t='<table><tr><th>channel</th><th>% poses</th><th>bottleneck Å</th><th>length Å</th><th>start-stable</th></tr>';
    p.channels.forEach(c=>{t+=`<tr><td>cl${c.cluster}</td><td>${fmt(100*c.freq,0)}</td><td>${fmt(c.bottleneck,2)} ± ${fmt(c.bottleneck_sd,2)}</td><td>${fmt(c.length,1)}</td><td>${c.stable===null?'n/a':fmt(100*c.stable,0)+'%'}</td></tr>`;});
    col.insertAdjacentHTML('beforeend',t+'</table>');
    (p.notes||[]).forEach(n=>col.insertAdjacentHTML('beforeend',`<div class="note">NOTE: ${n}</div>`));
    row.appendChild(col);
    const v=$3Dmol.createViewer(vd,{backgroundColor:'white'});
    v.addModel(p.protein,'pdb');
    for(const [name,txt] of Object.entries(p.ligands)){
      const m=v.addModel(txt,'pdb');
      if(name==='metal') m.setStyle({},{sphere:{radius:0.6,color:'0x228B22'}});
      else if(name==='substrate') m.setStyle({},{stick:{radius:0.22,colorscheme:'grayCarbon'},sphere:{scale:0.25,colorscheme:'grayCarbon'}});
      else m.setStyle({},{stick:{radius:0.14,colorscheme:'grayCarbon'}});
    }
    p.tunnels.forEach(tn=>tn.spheres.forEach(s=>{
      const sp=v.addSphere({center:{x:s[0],y:s[1],z:s[2]},radius:s[3],color:s[4],opacity:0.55});
    }));
    p.tunnels.forEach(tn=>{const s=tn.spheres[tn.spheres.length-1]; if(s) v.addLabel('cl'+tn.cluster,
      {position:{x:s[0],y:s[1],z:s[2]},fontSize:12,fontColor:'black',backgroundColor:'white',backgroundOpacity:0.7});});
    viewers.push(v);
  });
  style(); viewers.forEach(v=>{v.zoomTo({model:0}); v.render();});
  if(document.getElementById('sync').checked && viewers.length>1) link();
}
function style(){
  viewers.forEach(v=>{
    const m=v.getModel(0); m.setStyle({},document.getElementById('cart').checked?{cartoon:{color:'0xBBBBBB',opacity:0.75}}:{});
    v.removeAllSurfaces();
    if(document.getElementById('surf').checked) v.addSurface($3Dmol.SurfaceType.VDW,{opacity:0.35,color:'0xDDDDDD'},{model:m});
    v.render();
  });
}
function link(){
  viewers.forEach((v,i)=>v.setViewChangeCallback(()=>{ if(!document.getElementById('sync').checked) return;
    const view=v.getView(); viewers.forEach((w,j)=>{ if(j!==i){ w.setView(view); w.render(); } }); }));
}
sel.onchange=draw; ['cart','surf'].forEach(id=>document.getElementById(id).onchange=style);
document.getElementById('sync').onchange=draw;
draw();
</script></body></html>
"""


def build(entries, out, title="Substrate channels (CAVER)"):
    """entries = [(name, [(panel_label, run_dir), ...]), ...] -> one HTML at `out`."""
    data = [{"name": name, "panels": [panel(d, lab) for lab, d in panels]} for name, panels in entries]
    stops = [(rc.RADIUS_VMIN + (rc.RADIUS_VMAX - rc.RADIUS_VMIN) * i / 10) for i in range(11)]
    grad = ",".join("#%02x%02x%02x" % tuple(int(255 * v) for v in c) for c in rc._sphere_colors(stops))
    html = (PAGE.replace("__TITLE__", title).replace("__GRAD__", grad)
            .replace("__VMIN__", str(rc.RADIUS_VMIN)).replace("__VMAX__", str(rc.RADIUS_VMAX))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    with open(out, "w") as fh:
        fh.write(html)
    return out


def _cli(argv=None):
    ap = argparse.ArgumentParser(prog="channels_html", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Substrate channels (CAVER)")
    ap.add_argument("--entry", nargs="+", action="append", required=True, metavar="NAME_OR_LABEL=RUN_DIR",
                    help='enzyme name, then one or more "panel label=substrate_channels run dir"')
    a = ap.parse_args(argv)
    entries = []
    for e in a.entry:
        name, rest = e[0], e[1:]
        panels = [tuple(x.split("=", 1)) for x in rest]
        if not panels or any(len(p) != 2 for p in panels):
            ap.error(f"--entry {name}: give panels as 'label=run_dir'")
        entries.append((name, panels))
    out = build(entries, a.out, a.title)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
