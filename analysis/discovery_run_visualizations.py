"""Render retrospective discovery accounting without importing the discovery engine.

All biological names and family assignments come from the supplied run artifacts.
This analysis utility is not a production discovery input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#E69F00", "#009E73", "#697586"


def load(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def family_id(metadata):
    value = metadata.get("interpro_family_id") or metadata.get("screening_family_id")
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected scalar family ID; inspect this run's schema")
    return value


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", facecolor="white", bbox_inches="tight")
    svg_path = out / f"{name}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def table(headers, rows, element_id="", searchable=False):
    attrs = f' id="{element_id}"' if element_id else ""
    heads = "".join(f'<th tabindex="0">{html.escape(str(x))}</th>' for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows)
    search = f'<label>Filter candidates <input id="search" type="search" placeholder="Name, accession, family, organism…"></label><p id="visible-count" aria-live="polite"></p>' if searchable else ""
    return search + f'<div class="table-wrap"><table{attrs}><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = ["manifest.json", "ranked_candidates.csv", "candidate_coverage.csv",
               "agent_state/token_usage.json", "agent_state/candidate_registry.json",
               "agent_state/budget_checkpoint.json", "agent_state/mechanism_families.json",
               "agent_state/family_landscape.json", "logs/events.jsonl"]
    source_hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources}
    manifest = load(root / "manifest.json")
    usage = load(root / "agent_state/token_usage.json")
    registry = load(root / "agent_state/candidate_registry.json")["candidates"]
    checkpoint = load(root / "agent_state/budget_checkpoint.json")
    ranked = sorted(read_rows(root / "ranked_candidates.csv"), key=lambda x: int(x["rank"]))
    coverage = read_rows(root / "candidate_coverage.csv")
    events = [json.loads(line) for line in (root / "logs/events.jsonl").read_text().splitlines() if line.strip()]
    calls = usage["calls"]
    start = datetime.fromisoformat(manifest["timing"]["started_at"])
    wall = manifest["timing"]["wall_clock_s"]
    title = manifest["query"].splitlines()[0]
    statuses = Counter((v.get("statuses") or [{"status": "unknown"}])[-1]["status"] for v in registry.values())
    tiers = Counter(x["tier_label"] for x in ranked)
    target_evidence = Counter(x["target_activity_evidence"] for x in ranked)
    ranked_labels = Counter(x["interpro_family"] for x in ranked if x["interpro_family"])
    mechanism_labels = Counter(x["mechanism_family"] for x in ranked if x["mechanism_family"])
    by_accession = defaultdict(list)
    by_label = defaultdict(set)
    families = {}
    for key, entry in registry.items():
        meta = entry.get("metadata") or {}
        by_accession[entry.get("accession", key)].append(entry)
        fid = family_id(meta)
        if not fid:
            continue
        row = families.setdefault(fid, {"family_id": fid, "family_name": "", "registered_records": 0,
                                       "latest_evaluated_records": 0, "ranked_candidates": 0,
                                       "annotation_records": 0, "screening_only_records": 0})
        row["registered_records"] += 1
        row["latest_evaluated_records"] += bool(entry.get("statuses") and entry["statuses"][-1]["status"] == "evaluated")
        row["annotation_records"] += bool(meta.get("interpro_family_id"))
        row["screening_only_records"] += not bool(meta.get("interpro_family_id"))
        if meta.get("interpro_family"):
            row["family_name"] = meta["interpro_family"]
            by_label[meta["interpro_family"]].add(fid)
    for row in families.values():
        if not row["family_name"]:
            row["family_name"] = "Name not recorded (screening assignment)"
    for candidate in ranked:
        ids = {family_id(x.get("metadata") or {}) for x in by_accession[candidate["accession"]]}
        ids.discard(None)
        if not ids:
            ids = by_label[candidate["interpro_family"]]
        if len(ids) != 1:
            raise ValueError(f"Ambiguous ranked family for {candidate['accession']}: {ids}")
        candidate["family_id"] = next(iter(ids))
        families[candidate["family_id"]]["ranked_candidates"] += 1
    family_rows = sorted(families.values(), key=lambda x: (-x["ranked_candidates"], -x["registered_records"], x["family_id"]))
    stage_rows = defaultdict(lambda: Counter())
    for call in calls:
        stage = "controller" if isinstance(call["iteration"], int) else str(call["iteration"])
        stage_rows[stage]["calls"] += 1
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            stage_rows[stage][key] += call[key]
    stages = [{"stage": key, **value} for key, value in stage_rows.items()]
    token_fields = ["input_tokens", "cache_creation_tokens", "cache_read_tokens", "completion_tokens"]
    token_totals = [sum(x.get(key, 0) for x in calls) for key in token_fields]
    llm_events = [x for x in events if x["type"] == "llm_call"]
    checks = {
        "calls_match_events": len(calls) == len(llm_events),
        "request_ids_match": {x["request_id"] for x in calls} == {x["request_id"] for x in llm_events},
        "token_sum_matches": sum(x["total_tokens"] for x in calls) == usage["total_tokens"],
        "token_components_match": sum(token_totals) == usage["total_tokens"],
        "prompt_components_match": sum(token_totals[:3]) == usage["total_prompt_tokens"],
        "coverage_matches_registry": len(coverage) == len(registry),
        "ranked_accessions_unique": len({x["accession"] for x in ranked}) == len(ranked),
        "ranked_families_partition": sum(x["ranked_candidates"] for x in family_rows) == len(ranked),
        "source_files_unchanged": all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in source_hashes.items()),
    }
    if not all(checks.values()):
        raise ValueError(f"Accounting validation failed: {checks}")
    seconds = round(wall)
    duration = f"{seconds // 3600} h {(seconds % 3600) // 60:02d} min {seconds % 60:02d} s"
    metric_rows = [
        ("Discovery wall time", duration, "Manifest start → finish; later modeling excluded"),
        ("Started", manifest["timing"]["started_at"], "Timestamp includes UTC offset"),
        ("Finished", manifest["timing"]["finished_at"], "Timestamp includes UTC offset"),
        ("Run status", manifest["status"], "Resumable; token budget reached"),
        ("Model", usage["model"], "As recorded by the run"),
        ("Total tokens", f"{usage['total_tokens']:,}", "Prompt + completion, including cached prompt reads"),
        ("Prompt tokens", f"{usage['total_prompt_tokens']:,}", "New input + cache creation + cache reads"),
        ("Completion tokens", f"{usage['total_completion_tokens']:,}", "Recorded model output"),
        ("New input tokens", f"{token_totals[0]:,}", "Uncached input category; cache creation reported separately"),
        ("Cache creation tokens", f"{token_totals[1]:,}", "Included in prompt tokens"),
        ("Cache read tokens", f"{token_totals[2]:,}", f"{100 * token_totals[2] / usage['total_prompt_tokens']:.2f}% of prompt tokens"),
        ("Model calls", f"{len(calls):,}", f"Unmetered calls: {usage['unmetered_calls']}; accounting complete: {usage['accounting_complete']}"),
        ("Controller iterations used", str(manifest["result"]["iterations_used"]), "Manifest counter; distinct from model-call count"),
        ("Token budget / overrun", f"{checkpoint['token_budget']:,} / {checkpoint['token_overrun']:,}", "Final call crossed the budget"),
        ("Registered lead records", f"{len(registry):,}", "Registry records; not deduplicated proteins or validated hits"),
        ("Evaluation operations", str(manifest["result"]["candidates_evaluated"]), "Manifest result counter; not final status count"),
        ("Latest status: evaluated", str(statuses["evaluated"]), "Final registry snapshot"),
        ("Pending checkpoint entries", str(len(checkpoint["pending_candidates"])), "Checkpoint scope; not derived from registry status subtraction"),
        ("Ranked candidates", str(len(ranked)), "Unique accessions in ranked_candidates.csv"),
        ("Ranked InterPro families", str(len(ranked_labels)), "Recorded family labels; no taxonomy normalization"),
        ("Registry family IDs", str(len(families)), "Primary annotated ID, otherwise screening ID; one per record"),
        ("Landscape fold IDs", str(len({x['fold_id'] for x in load(root / 'agent_state/family_landscape.json')})), "Initial exploration landscape; separate scope"),
        ("Mechanism hypotheses", str(len(load(root / 'agent_state/mechanism_families.json'))), "Reasoned hypotheses; not experimental coverage"),
        ("Ranked mechanism labels", str(len(mechanism_labels)), f"Recorded broad assignments; {sum(not x['mechanism_family'] for x in ranked)} candidates unassigned"),
    ]
    write_rows(out / "run_metrics.csv", [dict(zip(["metric", "value", "definition"], r)) for r in metric_rows])
    write_rows(out / "candidates.csv", ranked)
    write_rows(out / "family_coverage.csv", family_rows)
    write_rows(out / "token_stages.csv", stages)
    call_rows = [{"timestamp": e["ts"], "elapsed_minutes": (datetime.fromisoformat(e["ts"]) - start).total_seconds() / 60,
                  "stage": "controller" if isinstance(e["iteration"], int) else str(e["iteration"]),
                  **{k: e[k] for k in ["request_id", "prompt_tokens", "completion_tokens", "total_tokens", "cumulative_tokens"]}} for e in llm_events]
    write_rows(out / "token_calls.csv", call_rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax = axes[0, 0]
    ax.axis("off")
    summary = [("Discovery time", duration), ("Tokens / model calls", f"{usage['total_tokens']:,} / {len(calls)}"),
               ("Registered lead records", f"{len(registry):,}"), ("Ranked candidate accessions", str(len(ranked))),
               ("Families: registry / ranked", f"{len(families)} IDs / {len(ranked_labels)} labels"),
               ("Stop reason", "Token budget reached")]
    t = ax.table(cellText=summary, colLabels=["Metric", "Recorded result"], cellLoc="left", colLoc="left", bbox=[0, 0.08, 1, 0.86])
    t.auto_set_font_size(False); t.set_fontsize(10)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("#e1e7ed")
        if i == 0: cell.set_facecolor("#e8f2fa"); cell.set_text_props(weight="bold")
    ax.set_title("A  Run accounting", loc="left")
    ax = axes[0, 1]
    y = list(range(len(stages)))
    ax.barh(y, [x["prompt_tokens"] / 1e6 for x in stages], color=BLUE, label="Prompt (includes cache)")
    ax.barh(y, [x["completion_tokens"] / 1e6 for x in stages], left=[x["prompt_tokens"] / 1e6 for x in stages], color=ORANGE, label="Completion")
    ax.set_yticks(y, [f"{x['stage'].replace('_', ' ')}\n({x['calls']} calls)" for x in stages]); ax.invert_yaxis()
    ax.set_xlabel("Tokens (millions)"); ax.legend(fontsize=9, loc="upper right"); ax.set_title("B  Token use by stage", loc="left")
    ax = axes[1, 0]
    ax.step([0] + [x["elapsed_minutes"] for x in call_rows], [0] + [x["cumulative_tokens"] / 1e6 for x in call_rows], where="post", color=BLUE)
    ax.axhline(checkpoint["token_budget"] / 1e6, color=ORANGE, ls="--", label="Token budget")
    ax.set(xlabel="Elapsed discovery time (minutes)", ylabel="Cumulative tokens (millions)", xlim=(0, wall / 60), ylim=(0, usage["total_tokens"] / 1e6 * 1.08))
    ax.set_title("C  Recorded token trajectory", loc="left"); ax.legend(fontsize=9)
    ax = axes[1, 1]
    labels = list(tiers)
    ax.barh(labels, [tiers[x] for x in labels], color=[BLUE, GREEN, GRAY][:len(labels)])
    for i, label in enumerate(labels): ax.text(tiers[label] + .4, i, str(tiers[label]), va="center")
    ax.set_xlim(0, max(tiers.values()) * 1.15); ax.invert_yaxis(); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Ranked candidates"); ax.set_title("D  Exported portfolio tiers", loc="left")
    fig.suptitle(title + "\nDiscovery phase • recorded accounting and hypotheses", fontsize=15)
    save(fig, out, "overview")

    fig, axes = plt.subplots(1, 2, figsize=(16, 14), sharey=True, layout="constrained", gridspec_kw={"width_ratios": [1, 1]})
    y = list(range(len(family_rows)))
    labels = [f"{x['family_id']}  " + textwrap.fill(x["family_name"], 48) for x in family_rows]
    for ax, key, color, title_text in zip(axes, ["registered_records", "ranked_candidates"], [GRAY, BLUE], ["Registered lead records", "Ranked candidate accessions"]):
        values = [x[key] for x in family_rows]
        ax.barh(y, values, color=color)
        for i, value in enumerate(values):
            ax.annotate(str(value), (max(value, .8) if key == "registered_records" else value, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
        if key == "registered_records": ax.set_xscale("log"); ax.set_xlim(.8, max(values) * 2.6); ax.set_xlabel("Records (log scale)")
        else: ax.set_xlim(0, max(values) * 1.2); ax.set_xlabel("Unique ranked accessions"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title_text)
    axes[0].set_yticks(y, labels, fontsize=9); axes[0].invert_yaxis()
    fig.suptitle(f"Family coverage: {len(families)} registry IDs; {len(ranked_labels)} represented in the ranked portfolio\nOne primary annotated ID per registry record, otherwise its screening ID; recorded assignments retained", fontsize=13)
    save(fig, out, "family_coverage")

    fig, ax = plt.subplots(figsize=(13, 4.8), layout="constrained")
    token_names = ["New input", "Cache creation", "Cache read", "Completion"]
    bars = ax.barh(token_names, [x / 1e6 for x in token_totals], color=[GRAY, BLUE, GREEN, ORANGE])
    for bar, value in zip(bars, token_totals): ax.text(bar.get_width() + .02, bar.get_y() + bar.get_height()/2, f"{value:,}", va="center")
    ax.set_xlim(0, max(token_totals) / 1e6 * 1.2); ax.invert_yaxis(); ax.set_xlabel("Tokens (millions)")
    ax.set_title("Token accounting • prompt categories are mutually exclusive\nTotal includes cache reads; no monetary cost inferred")
    save(fig, out, "token_accounting")

    fig, ax = plt.subplots(figsize=(15, 8.5), layout="constrained")
    ax.axis("off")
    top = ranked[:15]
    cells = [[x["rank"], textwrap.fill(x["protein_name"], 48), x["accession"], x["family_id"], textwrap.fill(x["tier_label"], 22)] for x in top]
    t = ax.table(cellText=cells, colLabels=["Rank", "Enzyme / protein", "Accession", "Family ID", "Exported tier"], colWidths=[.05,.43,.15,.13,.24], cellLoc="left", colLoc="left", bbox=[0, .03, 1, .88])
    t.auto_set_font_size(False); t.set_fontsize(10)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("#e1e7ed")
        if i == 0: cell.set_facecolor("#e8f2fa"); cell.set_text_props(weight="bold")
        elif i % 2 == 0: cell.set_facecolor("#f6f8fa")
    ax.set_title(f"First {len(top)} of {len(ranked)} exported candidates\nOriginal discovery order; target activity is a mechanistic-transfer hypothesis", loc="left", pad=20)
    save(fig, out, "candidate_table")

    caveats = [
        "Accounting covers the original discovery phase only. Later structural modeling, GPU time, and visualization generation are excluded; their cost cannot be inferred from this token ledger.",
        "Token totals count prompt plus completion tokens, including repeated cache reads. They are not unique text tokens or monetary cost. Cache reads must not be added to the prompt total a second time.",
        "A registry record is a lead, not necessarily a unique sequence. The manifest evaluation counter, latest registry statuses, and pending-checkpoint list have different scopes and do not form a single funnel.",
        "Family coverage uses recorded InterPro IDs and family labels, which can describe domains, families, or superfamilies. The registry ID count does not establish an equal number of independent mechanisms or experimental coverage.",
        f"Recorded target-activity evidence counts: {dict(target_evidence)}. Native characterization and target activity are distinct. Recorded broad mechanism assignments are reproduced without reclassification and may be coarse.",
        "Deprioritized analogue entries are retained in the full table. Exported tier and target-assessment decision are separate fields; the original values are preserved in candidates.csv.",
    ]
    headers = ["Rank", "Enzyme / protein", "Accession", "Gene", "Organism", "InterPro family", "Family ID", "Tier", "Target evidence", "Native evidence", "Rationale / uncertainty"]
    candidate_rows = [[x["rank"],x["protein_name"],x["accession"],x["gene"],x["organism"],x["interpro_family"],x["family_id"],x["tier_label"],x["target_activity_evidence"],x["native_activity_evidence"],x["priority_rationale"] + " Uncertainties: " + x["weaknesses"]] for x in ranked]
    def svg(name):
        # Drop XML/DTD prologue for inline HTML.
        return "<svg" + (out / f"{name}.svg").read_text().split("<svg", 1)[1]
    sections = [f"<h1>{html.escape(title)}</h1><p>Discovery run • {html.escape(root.name)}<br>Source snapshot: {html.escape(manifest['timing']['finished_at'])}</p>",
                '<nav><a href="#metrics">Metrics</a> · <a href="#families">Families</a> · <a href="#candidates">Candidates</a> · <a href="#definitions">Definitions</a></nav>',
                svg("overview"), '<h2 id="metrics">Timing and token accounting</h2>', table(["Metric", "Value", "Definition"], metric_rows), svg("token_accounting"),
                "<h3>Model stages</h3>", table(list(stages[0]), [list(x.values()) for x in stages]),
                '<h2 id="families">Family coverage</h2>', svg("family_coverage"),
                table(list(family_rows[0]), [list(x.values()) for x in family_rows]),
                '<h2 id="candidates">Discovered candidate portfolio</h2><p>All exported candidates; click column headers to sort. Original rank is preserved in the first column. No new scientific ranking was calculated.</p>',
                table(headers, candidate_rows, "candidates-table", searchable=True),
                '<h2 id="definitions">Definitions and limits</h2><ul>' + "".join(f"<li>{html.escape(x)}</li>" for x in caveats) + "</ul>",
                "<p>Companion files: README.md, run_metrics.csv, candidates.csv, family_coverage.csv, token_stages.csv, token_calls.csv, summary.json, provenance.json, and PNG/SVG figure exports.</p>"]
    css = "body{font:16px/1.5 system-ui,sans-serif;color:#172b40;max-width:1480px;margin:36px auto;padding:0 24px;background:#fff}h1{font-size:30px}h2{margin-top:48px}nav a{color:#0072B2}svg{width:100%;height:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4ec}th{background:#e8f2fa;cursor:pointer;position:sticky;top:0}tr:nth-child(even){background:#f7f9fb}.table-wrap{overflow:auto;max-height:750px}#candidates-table{min-width:1950px}#candidates-table td:nth-child(2){min-width:220px}#candidates-table td:last-child{min-width:480px}input{padding:10px;width:min(480px,90%);font:inherit}li{margin:10px 0}@media print{.table-wrap{max-height:none;overflow:visible}nav,input{display:none}}"
    js = """const tbl=document.querySelector('#candidates-table');const body=tbl.tBodies[0];const count=document.querySelector('#visible-count');function update(){const q=document.querySelector('#search').value.toLowerCase();let n=0;[...body.rows].forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q);if(!r.hidden)n++});count.textContent=n+' of '+body.rows.length+' candidates shown'}document.querySelector('#search').addEventListener('input',update);[...tbl.tHead.rows[0].cells].forEach((h,i)=>{let asc=true;function sort(){const rows=[...body.rows];rows.sort((a,b)=>{let x=a.cells[i].textContent,y=b.cells[i].textContent;return (i===0?Number(x)-Number(y):x.localeCompare(y))*(asc?1:-1)});rows.forEach(r=>body.appendChild(r));[...h.parentElement.cells].forEach(c=>c.removeAttribute('aria-sort'));h.setAttribute('aria-sort',asc?'ascending':'descending');asc=!asc}h.addEventListener('click',sort);h.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();sort()}})});update();"""
    (out / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Discovery run visualization</title><style>' + css + '</style><body>' + "\n".join(sections) + '<script>' + js + '</script></body></html>')
    summary = {"run_id": root.name, "discovery_wall_seconds": wall, "total_tokens": usage["total_tokens"],
               "registered_lead_records": len(registry), "ranked_candidate_accessions": len(ranked),
               "registry_family_ids": len(families), "ranked_family_labels": len(ranked_labels),
               "registry_latest_status_counts": dict(statuses), "exported_tiers": dict(tiers),
               "ranked_target_evidence": dict(target_evidence), "ranked_mechanism_labels": dict(mechanism_labels),
               "checks": checks, "caveats": caveats}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({"run_path": str(root), "source_sha256": source_hashes,
              "generator": str(Path(__file__).resolve()), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "matplotlib_version": matplotlib.__version__, "scope": "Retrospective discovery-phase accounting; no new discovery or modeling"}, indent=2) + "\n")
    md = [f"# {title}", "", "Open [the self-contained dashboard](index.html) for the searchable candidate table and figures.", "", "## Timing, tokens and coverage", "", "| Metric | Value | Definition |", "|---|---|---|"]
    md += ["| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |" for row in metric_rows]
    md += ["", "## Figures", "", "- [Overview](overview.png) ([SVG](overview.svg))", "- [Family coverage](family_coverage.png) ([SVG](family_coverage.svg))", "- [Token accounting](token_accounting.png) ([SVG](token_accounting.svg))", "- [First 15 candidates](candidate_table.png) ([SVG](candidate_table.svg))", "", "## All ranked candidates", "", "Original export order; hypotheses rather than demonstrated target activity.", "", "| Rank | Enzyme / protein | Accession | Family ID | Tier |", "|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(v).replace("|", "\\|") for v in [x["rank"], x["protein_name"], x["accession"], x["family_id"], x["tier_label"]]) + " |" for x in ranked]
    md += ["", "## Definitions and limits", ""] + [f"- {x}" for x in caveats]
    md += ["", "## Reproduce", "", "Run from the repository root using Python with matplotlib and NumPy installed:", "", "```bash", f"OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python analysis/discovery_run_visualizations.py '{root}' --out '{out}'", "```", "", "All data and biological labels are read from the supplied run; the generator imports no discovery code. `provenance.json` records source hashes. `summary.json` records accounting checks. Plots are descriptive counts and provider accounting; no inferential error bars or significance tests apply.", ""]
    (out / "README.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
