#!/usr/bin/env python3
"""Reusable HTML report of evaluated candidates' NATIVE reactions / substrates.

Built to be re-run as the discovery workflow improves, so results across runs can
be compared systematically. Reads each run's ``ranked_candidates.csv`` and emits a
single self-contained HTML file (2D structures drawn inline with RDKit — no
external assets beyond Google Fonts), openable in any browser.

Usage
-----
  # one run -> detailed per-candidate cards
  python analysis/candidate_native_report.py RUN_DIR [--out report.html]

  # several runs -> comparison table (rank/tier per run) + native chemistry
  python analysis/candidate_native_report.py RUN_A RUN_B [...] --out compare.html

  # --artifact emits a skeleton-free fragment for publishing as an artifact
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except Exception:  # noqa: BLE001
    _RDKIT = False


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> dict:
    csv_path = run_dir / "ranked_candidates.csv"
    rows = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
    meta = {"run_id": run_dir.name, "query": "", "n": len(rows), "target_smiles": ""}
    tc_path = run_dir / "target_context.json"
    if tc_path.exists():
        try:
            tc = json.loads(tc_path.read_text())
            meta["query"] = tc.get("query", "")
            meta["target_smiles"] = tc.get("substrate_smiles", "")
        except (OSError, ValueError):
            pass
    best: dict[str, dict] = {}
    for r in rows:
        acc = r.get("accession", "").strip() or r.get("gene", "")
        try:
            rank = int(r.get("rank") or 10**9)
        except ValueError:
            rank = 10**9
        r["_rank"] = rank
        if acc not in best or rank < best[acc]["_rank"]:
            best[acc] = r
    meta["candidates"] = sorted(best.values(), key=lambda r: r["_rank"])
    return meta


# --------------------------------------------------------------------------- #
# Structure rendering
# --------------------------------------------------------------------------- #
def _mol_svg(smiles: str, w: int = 150, h: int = 112) -> str:
    if not _RDKIT:
        return ""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 1
    d.DrawMolecule(m)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg[svg.find("<svg"):]


def render_substrates(smiles_field: str, names_field: str, cap: int = 8) -> str:
    smis = [s.strip() for s in (smiles_field or "").replace(".", ";").split(";") if s.strip()]
    names = [n.strip() for n in (names_field or "").split(";")]
    if not smis:
        return '<p class="empty">no native-substrate structures</p>'
    cells = []
    for i, smi in enumerate(smis[:cap]):
        svg = _mol_svg(smi)
        name = html.escape(names[i]) if i < len(names) and names[i] else ""
        body = svg or f'<code class="bad">{html.escape(smi[:36])}</code>'
        cells.append(f'<figure class="mol">{body}<figcaption>{name}</figcaption></figure>')
    more = f'<span class="more">+{len(smis) - cap} more</span>' if len(smis) > cap else ""
    return f'<div class="mols">{"".join(cells)}{more}</div>'


def render_reactions(cat: str, cap: int = 3) -> str:
    rxns = [x.strip() for x in (cat or "").split(" | ") if x.strip()]
    if not rxns:
        return '<p class="empty">no native reaction recorded</p>'
    items = "".join(f"<li>{_fmt_rxn(r)}</li>" for r in rxns[:cap])
    more = f'<li class="more">+{len(rxns) - cap} more reactions</li>' if len(rxns) > cap else ""
    return f'<ol class="rxns">{items}{more}</ol>'


def _fmt_rxn(r: str) -> str:
    # emphasise the reaction arrow so the transformation reads at a glance
    r = html.escape(r)
    return r.replace(" = ", ' <span class="arrow">→</span> ')


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _tier_key(tier_label: str) -> str:
    t = (tier_label or "").lower()
    return "t1" if "strong" in t else "t2" if "worth" in t else "t3"


def _uniprot(acc: str) -> str:
    acc = html.escape(acc)
    return (f'<a class="acc" href="https://www.uniprot.org/uniprotkb/{acc}" '
            f'target="_blank" rel="noopener">{acc}</a>')


def _pct_hosted(v: str) -> int:
    try:
        num, den = v.split("/")
        return round(100 * float(num) / float(den))
    except Exception:  # noqa: BLE001
        return 0


def _pct_fg(v: str) -> int:
    try:
        return round(100 * float(v))
    except (TypeError, ValueError):
        return 0


def _bar(label: str, pct: int, value: str, kind: str) -> str:
    return (f'<div class="metric"><span class="ml">{label}</span>'
            f'<span class="track"><i class="fill {kind}" style="width:{pct}%"></i></span>'
            f'<span class="mv">{html.escape(value)}</span></div>')


# --------------------------------------------------------------------------- #
# Candidate card (single-run view)
# --------------------------------------------------------------------------- #
def candidate_card(c: dict) -> str:
    tier = _tier_key(c.get("tier_label", ""))
    gene = html.escape(c.get("gene") or (c.get("protein_name", "") or "?")[:26])
    hosted = c.get("substrates_hosted", "")
    fg = c.get("weakest_fg_match", "")
    ev = html.escape(c.get("evidence_level", "") or "—")
    direction = html.escape(c.get("direction", "") or "—")
    meta_bits = []
    for lbl, key in (("Organism", "organism"), ("EC", "ec_numbers"), ("Family", "interpro_family")):
        val = html.escape((c.get(key) or "").strip() or "—")
        meta_bits.append(f"<div><dt>{lbl}</dt><dd>{val}</dd></div>")
    seqlen = html.escape(c.get("sequence_length", "") or "—")
    return f"""
    <article class="card" data-tier="{tier}">
      <span class="stripe"></span>
      <div class="cbody">
        <div class="chead">
          <span class="rank">{int(c.get('rank') or 0):02d}</span>
          <div class="ident"><h3>{gene}</h3>{_uniprot(c.get('accession',''))}</div>
          <span class="chip tier">{html.escape(c.get('tier_label',''))}</span>
        </div>
        <dl class="meta">{''.join(meta_bits)}</dl>
        <div class="fit">
          {_bar('substrates hosted', _pct_hosted(hosted), hosted or '—', 'k1')}
          {_bar('weakest FG match', _pct_fg(fg), fg or '—', 'k2')}
          <div class="tags">
            <span class="tag">{direction}</span>
            <span class="tag">evidence: {ev}</span>
            <span class="tag">{seqlen} aa</span>
          </div>
        </div>
        <section class="block">
          <h4>Native reaction</h4>
          {render_reactions(c.get('catalytic_activities',''))}
        </section>
        <section class="block">
          <h4>Native substrates</h4>
          {render_substrates(c.get('native_substrate_smiles',''), c.get('native_substrates',''))}
        </section>
      </div>
    </article>"""


def _summary_strip(run: dict) -> str:
    cands = run["candidates"]
    tiers = Counter(_tier_key(c.get("tier_label", "")) for c in cands)
    ev = Counter((c.get("evidence_level", "") or "—").lower() for c in cands)
    hosted2 = sum(1 for c in cands if c.get("substrates_hosted", "") == "2/2")
    stats = [
        ("candidates", str(len(cands)), ""),
        ("strong leads", str(tiers.get("t1", 0)), "t1"),
        ("worth testing", str(tiers.get("t2", 0)), "t2"),
        ("host both substrates", str(hosted2), ""),
        ("high evidence", str(ev.get("high", 0)), ""),
    ]
    cells = "".join(
        f'<div class="stat {k}"><span class="num">{v}</span><span class="lab">{lab}</span></div>'
        for lab, v, k in stats)
    return f'<div class="summary">{cells}</div>'


def render_single(run: dict) -> str:
    cards = "\n".join(candidate_card(c) for c in run["candidates"])
    target = ""
    if run.get("target_smiles"):
        target = ('<div class="target"><span class="tlabel">Target</span>'
                  + render_substrates(run["target_smiles"], "", cap=6) + "</div>")
    return f"""
    <header class="hero">
      <p class="eyebrow">Enzyme discovery · evaluated candidates</p>
      <h1>{html.escape(run['query'] or run['run_id'])}</h1>
      <p class="runid">{html.escape(run['run_id'])}</p>
      {_summary_strip(run)}
      {target}
    </header>
    <main class="grid">{cards}</main>"""


# --------------------------------------------------------------------------- #
# Comparison view (multi-run)
# --------------------------------------------------------------------------- #
def render_comparison(runs: list[dict]) -> str:
    union: dict[str, dict] = {}
    for run in runs:
        for c in run["candidates"]:
            union.setdefault(c.get("accession", "") or c.get("gene", ""), c)
    rank_of = [{(c.get("accession") or c.get("gene")): c for c in r["candidates"]} for r in runs]

    def best_rank(acc):
        return min((ro[acc]["_rank"] for ro in rank_of if acc in ro), default=10**9)

    heads = "".join(f'<th class="runcol">{html.escape(r["run_id"][:10])}…</th>' for r in runs)
    body = []
    for acc in sorted(union, key=best_rank):
        c = union[acc]
        cells = []
        for ro in rank_of:
            if acc in ro:
                rc = ro[acc]
                cells.append(
                    f'<td class="rankcell" data-tier="{_tier_key(rc.get("tier_label",""))}">'
                    f'<span class="rk">#{rc.get("rank","")}</span>'
                    f'<span class="tl">{html.escape((rc.get("tier_label","") or "")[:12])}</span></td>')
            else:
                cells.append('<td class="absent">·</td>')
        body.append(f"""
        <tr>
          <th scope="row" class="cand">
            <span class="g">{html.escape(c.get('gene') or '?')}</span>
            {_uniprot(acc)}
            <span class="fam">{html.escape(c.get('interpro_family','') or '')}</span>
          </th>
          {''.join(cells)}
          <td class="chem">{render_reactions(c.get('catalytic_activities',''), cap=1)}
              {render_substrates(c.get('native_substrate_smiles',''), c.get('native_substrates',''), cap=5)}</td>
        </tr>""")
    return f"""
    <header class="hero">
      <p class="eyebrow">Enzyme discovery · cross-run comparison</p>
      <h1>{html.escape(runs[0]['query'] or 'Candidate comparison')}</h1>
      <p class="runid">{len(union)} distinct candidates across {len(runs)} runs · ranked by best rank</p>
    </header>
    <main class="tablewrap">
      <table class="cmp">
        <thead><tr><th class="cand">Candidate</th>{heads}<th class="chem">Native reaction &amp; substrates</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </main>"""


# --------------------------------------------------------------------------- #
# Shell / styling
# --------------------------------------------------------------------------- #
_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&'
    'family=IBM+Plex+Sans:wght@400;500;600&'
    'family=IBM+Plex+Mono:wght@400;500&display=swap">'
)

_CSS = """
:root{
 --bg:#f4f6f8; --card:#ffffff; --panel:#f0f3f7; --fg:#161b22; --muted:#5b6672;
 --line:#e2e7ed; --accent:#2f57d6; --accent-soft:#e7edfb;
 --t1:#0f8a4f; --t2:#b06a00; --t3:#7a4fd0;
 --t1-soft:#e3f5ea; --t2-soft:#f8efdc; --t3-soft:#efe8fb;
 --sans:"IBM Plex Sans",system-ui,sans-serif;
 --serif:"Fraunces",Georgia,serif;
 --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
 --radius:14px; --shadow:0 1px 2px rgba(20,27,34,.05),0 6px 20px rgba(20,27,34,.05);}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#0d1117; --card:#161c24; --panel:#12171e; --fg:#e6ebf1; --muted:#9aa4b0;
 --line:#262d37; --accent:#7b9bff; --accent-soft:#1b2540;
 --t1:#41c07a; --t2:#e0a53c; --t3:#a583f2;
 --t1-soft:#122a1e; --t2-soft:#2c2413; --t3-soft:#231c39;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.35);}}
:root[data-theme=dark]{
 --bg:#0d1117; --card:#161c24; --panel:#12171e; --fg:#e6ebf1; --muted:#9aa4b0;
 --line:#262d37; --accent:#7b9bff; --accent-soft:#1b2540;
 --t1:#41c07a; --t2:#e0a53c; --t3:#a583f2;
 --t1-soft:#122a1e; --t2-soft:#2c2413; --t3-soft:#231c39;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.35);}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
 -webkit-font-smoothing:antialiased;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px;padding-block:40px}

/* header */
.hero{margin-bottom:28px}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--accent);font-weight:600;margin:0 0 10px}
.hero h1{font-family:var(--serif);font-weight:600;font-size:clamp(26px,4vw,40px);
 line-height:1.1;margin:0;text-wrap:balance;letter-spacing:-.01em;max-width:22ch}
.runid{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin:10px 0 0}

.summary{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:12px 16px;min-width:104px;box-shadow:var(--shadow)}
.stat .num{display:block;font-family:var(--serif);font-weight:600;font-size:26px;
 line-height:1;font-variant-numeric:tabular-nums}
.stat .lab{display:block;font-size:11.5px;color:var(--muted);margin-top:5px;
 text-transform:uppercase;letter-spacing:.04em}
.stat.t1 .num{color:var(--t1)} .stat.t2 .num{color:var(--t2)}

.target{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:20px;
 padding:12px 16px;background:var(--accent-soft);border-radius:12px}
.target .tlabel{font-size:11px;letter-spacing:.1em;text-transform:uppercase;
 font-weight:600;color:var(--accent)}

/* card grid */
.grid{display:grid;gap:18px;grid-template-columns:repeat(auto-fill,minmax(360px,1fr))}
.card{position:relative;background:var(--card);border:1px solid var(--line);
 border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow);
 transition:transform .15s ease,box-shadow .15s ease}
.card:hover{transform:translateY(-2px);box-shadow:0 4px 10px rgba(20,27,34,.08),0 16px 40px rgba(20,27,34,.10)}
.card .stripe{position:absolute;inset:0 auto 0 0;width:4px}
.card[data-tier=t1] .stripe{background:var(--t1)}
.card[data-tier=t2] .stripe{background:var(--t2)}
.card[data-tier=t3] .stripe{background:var(--t3)}
.cbody{padding:16px 18px 18px 20px}

.chead{display:flex;align-items:center;gap:12px}
.rank{font-family:var(--mono);font-size:13px;color:var(--muted);
 font-variant-numeric:tabular-nums}
.ident{margin-right:auto;min-width:0}
.ident h3{font-family:var(--serif);font-weight:600;font-size:20px;margin:0;
 line-height:1.15;letter-spacing:-.01em}
a.acc{font-family:var(--mono);font-size:12px;color:var(--accent);text-decoration:none}
a.acc:hover{text-decoration:underline}
.chip.tier{font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;white-space:nowrap}
[data-tier=t1] .chip.tier{background:var(--t1-soft);color:var(--t1)}
[data-tier=t2] .chip.tier{background:var(--t2-soft);color:var(--t2)}
[data-tier=t3] .chip.tier{background:var(--t3-soft);color:var(--t3)}

.meta{display:grid;grid-template-columns:1fr;gap:2px;margin:14px 0 0}
.meta div{display:flex;gap:8px;font-size:12.5px;min-width:0}
.meta dt{color:var(--muted);flex:0 0 62px;text-transform:uppercase;font-size:10.5px;
 letter-spacing:.04em;padding-top:2px}
.meta dd{margin:0;min-width:0;overflow:hidden;text-overflow:ellipsis}

.fit{margin:16px 0 4px;padding:14px 0;border-top:1px solid var(--line);
 border-bottom:1px solid var(--line)}
.metric{display:flex;align-items:center;gap:10px;margin:0 0 8px}
.metric .ml{flex:0 0 118px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.track{flex:1;height:6px;background:var(--panel);border-radius:6px;overflow:hidden}
.fill{display:block;height:100%;border-radius:6px}
.fill.k1{background:var(--accent)} .fill.k2{background:var(--t2)}
.metric .mv{flex:0 0 34px;text-align:right;font-family:var(--mono);font-size:12px;
 font-variant-numeric:tabular-nums}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
.tag{font-size:11px;color:var(--muted);background:var(--panel);border-radius:6px;padding:2px 9px}

.block{margin-top:16px}
.block h4{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;
 color:var(--muted);margin:0 0 8px;font-weight:600}
.rxns{margin:0;padding:0;list-style:none;font-family:var(--mono);font-size:12px;line-height:1.5}
.rxns li{padding:8px 12px;background:var(--panel);border-radius:8px;margin-bottom:6px;
 word-break:break-word}
.rxns .arrow{color:var(--accent);font-weight:600;padding:0 2px}
.rxns .more,.more{font-family:var(--sans);font-style:italic;color:var(--muted);background:none;padding-left:2px}

.mols{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}
.mol{margin:0;width:150px;text-align:center;background:var(--panel);
 border:1px solid var(--line);border-radius:10px;padding:6px 6px 8px;
 transition:border-color .15s}
.mol:hover{border-color:var(--accent)}
.mol svg{display:block;margin:0 auto;max-width:100%}
.mol figcaption{font-size:10px;color:var(--muted);margin-top:4px;line-height:1.3;word-break:break-word}
.bad{font-family:var(--mono);font-size:10px;color:var(--muted)}
.empty{font-size:12px;color:var(--muted);font-style:italic;margin:0}

/* comparison table */
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:var(--radius);
 background:var(--card);box-shadow:var(--shadow)}
table.cmp{border-collapse:collapse;width:100%;min-width:720px}
.cmp th,.cmp td{padding:12px 14px;text-align:left;vertical-align:top;
 border-bottom:1px solid var(--line)}
.cmp thead th{position:sticky;top:0;background:var(--card);z-index:1;font-size:11px;
 text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
.cmp .cand{min-width:150px} .cmp .cand .g{font-family:var(--serif);font-weight:600;
 font-size:16px;display:block} .cmp .cand .fam{display:block;font-size:11px;color:var(--muted);margin-top:3px}
.cmp .rankcell{white-space:nowrap;font-variant-numeric:tabular-nums}
.cmp .rankcell .rk{font-family:var(--mono);font-weight:500}
.cmp .rankcell .tl{display:block;font-size:10.5px;color:var(--muted)}
.cmp .rankcell[data-tier=t1] .rk{color:var(--t1)} .cmp .rankcell[data-tier=t2] .rk{color:var(--t2)}
.cmp .rankcell[data-tier=t3] .rk{color:var(--t3)}
.cmp .absent{color:var(--line);text-align:center;font-size:18px}
.cmp .chem{min-width:340px}
.cmp .chem .rxns{margin-bottom:8px}

@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media (max-width:520px){.grid{grid-template-columns:1fr}.meta dt{flex-basis:54px}}
"""


def build_html(runs: list[dict], fragment: bool = False) -> str:
    title = "Candidate Chemistry"
    inner = render_comparison(runs) if len(runs) > 1 else render_single(runs[0])
    note = "" if _RDKIT else '<p class="empty">RDKit unavailable — structures not drawn.</p>'
    content = f'{note}<div class="wrap">{inner}</div>'
    head = f"<title>{title}</title>{_FONTS}<style>{_CSS}</style>"
    if fragment:
        return head + "\n" + content
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"{head}</head><body>{content}</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="one or more run directories")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--artifact", action="store_true",
                    help="emit a skeleton-free fragment for publishing as an artifact")
    args = ap.parse_args()

    runs = []
    for rd in args.runs:
        p = Path(rd)
        if not (p / "ranked_candidates.csv").exists():
            print(f"skip {rd}: no ranked_candidates.csv", file=sys.stderr)
            continue
        runs.append(load_run(p))
    if not runs:
        sys.exit("no runs with ranked_candidates.csv")

    out = Path(args.out) if args.out else Path(args.runs[0]) / (
        "candidate_comparison.html" if len(runs) > 1 else "candidate_native_report.html")
    out.write_text(build_html(runs, fragment=args.artifact))
    print(f"wrote {out}  ({len(runs)} run(s), {sum(r['n'] for r in runs)} candidate rows)")


if __name__ == "__main__":
    main()
