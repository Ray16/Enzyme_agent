#!/usr/bin/env python3
"""Reusable HTML report of evaluated candidates' NATIVE reactions / substrates.

Built to be re-run as the discovery workflow improves, so results across runs can
be compared systematically. Reads each run's ``ranked_candidates.csv`` and emits a
single self-contained HTML file (2D structures drawn inline with RDKit — no
external assets), openable in any browser.

Usage
-----
  # one run -> detailed per-candidate cards
  python analysis/candidate_native_report.py RUN_DIR [--out report.html]

  # several runs -> comparison table (rank/tier per run) + native chemistry
  python analysis/candidate_native_report.py RUN_A RUN_B [RUN_C ...] --out compare.html

Each candidate shows: rank, tier, gene/accession (UniProt link), organism, EC,
InterPro family, direction, the decomposed substrate-fit axes, evidence level,
its NATIVE REACTION(s) (from catalytic_activities), and the 2D structures of its
native substrates (from native_substrate_smiles).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
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
    meta = {"run_id": run_dir.name, "query": "", "n": len(rows)}
    summ = run_dir / "target_context.json"
    if summ.exists():
        try:
            tc = json.loads(summ.read_text())
            meta["query"] = tc.get("query", "")
            meta["target_smiles"] = tc.get("substrate_smiles", "")
        except (OSError, ValueError):
            pass
    # dedupe by accession keeping best (lowest) rank — the CSV is already the
    # decision list, but guard against any repeats.
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
def _mol_svg(smiles: str, w: int = 150, h: int = 110) -> str:
    if not _RDKIT:
        return ""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = d.drawOptions()
    opts.clearBackground = False
    d.DrawMolecule(m)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    # strip the XML prolog so it embeds inline cleanly
    return svg[svg.find("<svg"):]


def render_substrates(smiles_field: str, names_field: str) -> str:
    smis = [s.strip() for s in (smiles_field or "").replace(".", ";").split(";") if s.strip()]
    names = [n.strip() for n in (names_field or "").split(";")]
    if not smis:
        return '<div class="nostruct">no native-substrate SMILES</div>'
    cells = []
    for i, smi in enumerate(smis[:6]):
        svg = _mol_svg(smi)
        name = html.escape(names[i]) if i < len(names) else ""
        body = svg or f'<code class="badsmiles">{html.escape(smi[:40])}</code>'
        cells.append(f'<figure class="mol">{body}<figcaption>{name}</figcaption></figure>')
    more = f'<span class="more">+{len(smis) - 6} more</span>' if len(smis) > 6 else ""
    return f'<div class="mols">{"".join(cells)}{more}</div>'


def render_reactions(cat: str) -> str:
    rxns = [x.strip() for x in (cat or "").split(" | ") if x.strip()]
    if not rxns:
        return '<div class="norxn">no native reaction recorded</div>'
    items = "".join(f"<li>{html.escape(r)}</li>" for r in rxns[:4])
    more = f'<li class="more">+{len(rxns) - 4} more reactions</li>' if len(rxns) > 4 else ""
    return f'<ol class="rxns">{items}{more}</ol>'


# --------------------------------------------------------------------------- #
# HTML fragments
# --------------------------------------------------------------------------- #
def _uniprot_link(acc: str) -> str:
    acc = html.escape(acc)
    return f'<a href="https://www.uniprot.org/uniprotkb/{acc}" target="_blank">{acc}</a>'


def _tier_class(tier_label: str) -> str:
    t = (tier_label or "").lower()
    if "strong" in t:
        return "t1"
    if "worth" in t:
        return "t2"
    return "t3"


def candidate_card(c: dict) -> str:
    gene = html.escape(c.get("gene") or c.get("protein_name", "")[:28] or "?")
    org = html.escape(c.get("organism", ""))
    ec = html.escape(c.get("ec_numbers", ""))
    fam = html.escape(c.get("interpro_family", ""))
    tier = html.escape(c.get("tier_label", ""))
    direction = html.escape(c.get("direction", ""))
    hosted = html.escape(c.get("substrates_hosted", ""))
    fg = html.escape(c.get("weakest_fg_match", ""))
    ev = html.escape(c.get("evidence_level", ""))
    seqlen = html.escape(c.get("sequence_length", ""))
    return f"""
    <div class="card {_tier_class(c.get('tier_label',''))}">
      <div class="chead">
        <span class="rank">#{c.get('rank','')}</span>
        <span class="gene">{gene}</span>
        <span class="acc">{_uniprot_link(c.get('accession',''))}</span>
        <span class="tier">{tier}</span>
      </div>
      <div class="meta">
        <span>{org}</span><span>EC {ec or '—'}</span><span>{fam or '—'}</span>
      </div>
      <div class="badges">
        <span class="b">dir: {direction}</span>
        <span class="b">hosts {hosted}</span>
        <span class="b">fg-match {fg}</span>
        <span class="b">evidence {ev}</span>
        <span class="b">len {seqlen}</span>
      </div>
      <div class="section-label">Native reaction</div>
      {render_reactions(c.get('catalytic_activities',''))}
      <div class="section-label">Native substrates</div>
      {render_substrates(c.get('native_substrate_smiles',''), c.get('native_substrates',''))}
    </div>"""


def render_single(run: dict) -> str:
    cards = "\n".join(candidate_card(c) for c in run["candidates"])
    target = ""
    if run.get("target_smiles"):
        target = ('<div class="target"><b>Target substrate(s):</b> '
                  + render_substrates(run["target_smiles"], "") + "</div>")
    return f"""
    <header>
      <h1>{html.escape(run['query'] or run['run_id'])}</h1>
      <p class="sub">{html.escape(run['run_id'])} · {run['n']} evaluated candidates</p>
      {target}
    </header>
    <main>{cards}</main>"""


def render_comparison(runs: list[dict]) -> str:
    # union of candidates by accession; native chemistry from the first run that has it
    union: dict[str, dict] = {}
    for run in runs:
        for c in run["candidates"]:
            acc = c.get("accession", "") or c.get("gene", "")
            union.setdefault(acc, c)
    run_ids = [r["run_id"] for r in runs]
    rank_of = [{(c.get("accession") or c.get("gene")): c for c in r["candidates"]} for r in runs]

    def best_rank(acc):
        return min((ro[acc]["_rank"] for ro in rank_of if acc in ro), default=10**9)

    header_cols = "".join(f"<th>{html.escape(rid[:22])}</th>" for rid in run_ids)
    body = []
    for acc in sorted(union, key=best_rank):
        c = union[acc]
        gene = html.escape(c.get("gene") or "?")
        cells = []
        for ro in rank_of:
            if acc in ro:
                rc = ro[acc]
                cells.append(f'<td class="{_tier_class(rc.get("tier_label",""))}">'
                             f'#{rc.get("rank","")}<br><small>{html.escape(rc.get("tier_label",""))}</small></td>')
            else:
                cells.append('<td class="absent">—</td>')
        rxn = render_reactions(c.get("catalytic_activities", ""))
        subs = render_substrates(c.get("native_substrate_smiles", ""), c.get("native_substrates", ""))
        body.append(f"""
        <tr>
          <td class="cand"><b>{gene}</b><br>{_uniprot_link(acc)}<br>
              <small>{html.escape(c.get('interpro_family',''))}</small></td>
          {''.join(cells)}
          <td class="chem">{rxn}{subs}</td>
        </tr>""")
    q = html.escape(runs[0]["query"] or "candidate comparison")
    return f"""
    <header>
      <h1>Candidate comparison — native chemistry</h1>
      <p class="sub">{q}<br>{len(union)} distinct candidates across {len(runs)} runs</p>
    </header>
    <table class="cmp">
      <thead><tr><th>Candidate</th>{header_cols}<th>Native reaction &amp; substrates</th></tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>"""


_CSS = """
:root{--bg:#f6f6f4;--fg:#1a1a1a;--muted:#5b6167;--card:#ffffff;--line:#e3e3e3;
 --t1:#1a7f37;--t2:#9a6700;--t3:#8250df;--accent:#0969da;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--card:#161b22;--line:#30363d;
 --t1:#3fb950;--t2:#d29922;--t3:#a371f7;--accent:#4493f8;}}
:root[data-theme=dark]{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--card:#161b22;
 --line:#30363d;--t1:#3fb950;--t2:#d29922;--t3:#a371f7;--accent:#4493f8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 padding:0 16px}
header{max-width:1100px;margin:24px auto 8px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 12px}
main{max-width:1100px;margin:0 auto;display:grid;gap:14px;
 grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-left-width:4px;
 border-radius:8px;padding:12px 14px}
.card.t1{border-left-color:var(--t1)} .card.t2{border-left-color:var(--t2)}
.card.t3{border-left-color:var(--t3)}
.chead{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.rank{font-weight:700;color:var(--muted)} .gene{font-weight:700;font-size:16px}
.acc a{color:var(--accent);text-decoration:none} .tier{margin-left:auto;color:var(--muted);font-size:12px}
.meta{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:4px 0}
.badges{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
.b{font-size:11px;background:color-mix(in srgb,var(--accent) 12%,transparent);
 border:1px solid var(--line);border-radius:10px;padding:1px 8px}
.section-label{font-size:11px;text-transform:uppercase;letter-spacing:.04em;
 color:var(--muted);margin:10px 0 3px}
.rxns{margin:0;padding-left:18px;font-size:12.5px} .rxns li{margin:2px 0}
.rxns .more,.more{color:var(--muted);font-style:italic}
.mols{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end}
.mol{margin:0;text-align:center} .mol svg{background:transparent}
.mol figcaption{font-size:10px;color:var(--muted);max-width:150px;word-break:break-word}
.badsmiles{font-size:10px;color:var(--muted)} .nostruct,.norxn{color:var(--muted);font-size:12px}
.target{background:var(--card);border:1px dashed var(--line);border-radius:8px;padding:8px 12px;margin:8px 0}
table.cmp{max-width:1200px;margin:0 auto;border-collapse:collapse;width:100%}
.cmp th,.cmp td{border:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}
.cmp th{background:var(--card);position:sticky;top:0}
.cmp td.t1{color:var(--t1);font-weight:700} .cmp td.t2{color:var(--t2);font-weight:700}
.cmp td.absent{color:var(--muted);text-align:center} .cmp td.cand{min-width:150px}
.cmp td.chem{min-width:340px}
@media(max-width:640px){main{grid-template-columns:1fr}}
"""


def build_html(runs: list[dict], fragment: bool = False) -> str:
    # Stable, gallery-friendly name (no "name — explainer"); the specifics live
    # in the run header and the publish description.
    title = "Candidate Chemistry"
    inner = render_comparison(runs) if len(runs) > 1 else render_single(runs[0])
    rdkit_note = "" if _RDKIT else ('<p class="sub" style="color:#b00">RDKit not '
                                    "available — structures not drawn.</p>")
    if fragment:
        # For publishing as an artifact: the host wraps the file in its own
        # doctype/head/body skeleton, so emit only title + style + content.
        return f"<title>{title}</title>\n<style>{_CSS}</style>\n{rdkit_note}{inner}"
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head>
<body>{rdkit_note}{inner}</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="one or more run directories")
    ap.add_argument("--out", default=None, help="output HTML path")
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

    out = Path(args.out) if args.out else Path(
        args.runs[0]) / ("candidate_comparison.html" if len(runs) > 1
                         else "candidate_native_report.html")
    out.write_text(build_html(runs))
    total = sum(r["n"] for r in runs)
    print(f"wrote {out}  ({len(runs)} run(s), {total} candidate rows)")


if __name__ == "__main__":
    main()
