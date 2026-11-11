"""Plot search development from a completed run's own retrospective evidence."""

from collections import Counter, defaultdict
from datetime import datetime
import json
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


BLUE, ORANGE, GREEN, GRAY, RED = "#0072B2", "#E69F00", "#009E73", "#8B96A5", "#D55E00"


def render_search_story(root, out, manifest, registry, ranked, families, events,
                        call_rows, save, write_rows):
    """Keep denominators explicit and validate every proposed subset relationship."""
    start = datetime.fromisoformat(manifest["timing"]["started_at"])
    end_minutes = manifest["timing"]["wall_clock_s"] / 60
    elapsed = lambda value: (datetime.fromisoformat(value) - start).total_seconds() / 60
    trace = [json.loads(line) for line in (root / "reasoning/trace.jsonl").read_text().splitlines() if line.strip()]
    reading = json.loads((root / "agent_state/literature_reading.json").read_text())
    checkpoint = json.loads((root / "agent_state/budget_checkpoint.json").read_text())
    assessed = [v for v in registry.values() if v.get("metadata", {}).get("target_assessment")]
    assessed_ids = {v["accession"] for v in assessed}
    exported_ids = {v["accession"] for v in ranked}
    decisions = Counter(v["metadata"]["target_assessment"]["decision"] for v in assessed)
    export_decisions = {v["accession"] for v in assessed
                        if v["metadata"]["target_assessment"]["decision"] in {"prioritize", "investigate"}}
    if len(assessed_ids) != len(assessed) or exported_ids != export_decisions:
        raise ValueError("The proposed review-to-export relationship needs manual review")
    first_reviews = {}
    for event in trace:
        if event["type"] == "target_assessment_recorded":
            accession = event["record"]["accession"]
            if accession in assessed_ids:
                first_reviews.setdefault(accession, elapsed(event["occurred_at"]))
    if set(first_reviews) != assessed_ids:
        raise ValueError("Some assessed records lack a timestamped review")
    lead_times = sorted(elapsed(v["first_seen"]) for v in registry.values())
    review_times = sorted(first_reviews.values())
    source_groups = {}
    source_counts = Counter()
    for key, entry in registry.items():
        source = entry.get("metadata", {}).get("source_tool", "")
        if source.startswith("family_substrate_fit:"):
            group = "Family substrate screening"
        elif source.startswith("search_"):
            group = "Direct database lookup"
        elif "literature" in source or source.startswith("fulltext_"):
            group = "Literature / full-text leads"
        else:
            group = "Other / unrecorded source"
        source_groups[key] = group
        source_counts[group] += 1
    tool_events = [event for event in events if event["type"] == "tool_call"]
    tool_names = sorted({event["tool"] for event in tool_events}, key=lambda name: (-sum(e["tool"] == name for e in tool_events), name))
    activity = [{"elapsed_minutes": elapsed(e["ts"]), "tool": e["tool"], "status": e["status"],
                 "result_count_as_recorded": e.get("count", ""), "duration_seconds": e.get("elapsed_s", ""),
                 "query_or_arguments": json.dumps(e.get("args", {}), ensure_ascii=False)} for e in tool_events]
    write_rows(out / "search_activity.csv", activity)
    research = [{"elapsed_minutes": elapsed(e["ts"]), "decision_id": e["decision_id"],
                 "action": e["action"], "revises": e.get("revises") or "", "hypothesis": e["hypothesis"],
                 "prediction": e["prediction"], "reasoning": e["reasoning"]}
                for e in events if e["type"] == "research_decision"]
    write_rows(out / "research_decisions.csv", research)
    download_counts = Counter(v.get("status", "unknown") for v in reading["downloads"].values())
    numeric_pairs = []
    targets = manifest["substrate"]["smiles"].split(".")
    for candidate in ranked:
        comparison = {x["target_smiles"]: x.get("analogue_fit")
                      for x in json.loads(candidate.get("substrate_partner_fits") or "[]")}
        if len(targets) == 2 and all(isinstance(comparison.get(t), (int, float)) for t in targets):
            numeric_pairs.append({"accession": candidate["accession"], "protein_name": candidate["protein_name"],
                                  "rank": int(candidate["rank"]), "tier": candidate["tier_label"],
                                  "partner_1_fit": comparison[targets[0]], "partner_2_fit": comparison[targets[1]]})
    if numeric_pairs:
        write_rows(out / "partner_comparisons.csv", numeric_pairs)
    family_outcomes = []
    for family in families:
        row = {"family_id": family["family_id"], "family_name": family["family_name"],
               "no_target_assessment": 0, "defer": 0, "exclude": 0, "exported": 0}
        family_outcomes.append(row)
    family_outcomes.append({"family_id": "Unassigned", "family_name": "No recorded family ID",
                            "no_target_assessment": 0, "defer": 0, "exclude": 0, "exported": 0})
    family_index = {row["family_id"]: row for row in family_outcomes}
    for entry in registry.values():
        meta = entry.get("metadata", {})
        fid = meta.get("interpro_family_id") or meta.get("screening_family_id") or "Unassigned"
        decision = meta.get("target_assessment", {}).get("decision", "no_target_assessment")
        bucket = "exported" if entry["accession"] in exported_ids else decision
        if bucket not in family_index[fid]:
            raise ValueError(f"Unmapped review outcome: {bucket}")
        family_index[fid][bucket] += 1
    outcome_keys = ["no_target_assessment", "defer", "exclude", "exported"]
    assert sum(sum(row[k] for k in outcome_keys) for row in family_outcomes) == len(registry)
    assert sum(row["exported"] for row in family_outcomes) == len(ranked)
    write_rows(out / "family_review_outcomes.csv", family_outcomes)
    timeline = sorted([(t, "lead_record") for t in lead_times] + [(t, "first_target_assessment") for t in review_times])
    cumulative = Counter()
    timeline_rows = []
    for minute, kind in timeline:
        cumulative[kind] += 1
        timeline_rows.append({"elapsed_minutes": minute, "event": kind,
                              "lead_records": cumulative["lead_record"], "target_assessed_records": cumulative["first_target_assessment"]})
    write_rows(out / "search_progress.csv", timeline_rows)

    def progression(ax):
        for times, color, name in [(lead_times, BLUE, "Registered leads"), (review_times, GREEN, "Target-assessed leads")]:
            ax.step([0] + times + [end_minutes], [0] + list(range(1, len(times) + 1)) + [len(times)], where="post", color=color, lw=2, label=name)
            ax.annotate(f"{len(times):,}", (end_minutes, len(times)), xytext=(-4, 7), textcoords="offset points", ha="right", color=color, weight="bold")
        ax.set(xlim=(0, end_minutes * 1.01), ylim=(0, len(registry) * 1.15), xlabel="Minutes from discovery start", ylabel="Cumulative lead records")
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16), layout="constrained", gridspec_kw={"height_ratios": [1, 1.15, 1]})
    fig.suptitle(manifest["query"].splitlines()[0] + "\nHow the search developed and where evidence remains incomplete"
                 + f"\n{end_minutes:.1f} minutes  ·  {call_rows[-1]['cumulative_tokens'] / 1e6:.2f}M tokens  ·  {len(registry):,} leads  ·  {len(assessed)} assessed  ·  {len(ranked)} candidates", fontsize=17, weight="bold")
    ax = axes[0, 0]; progression(ax); ax.set_title("A  Leads accumulated faster than target reviews", loc="left", pad=13)
    ax = axes[0, 1]
    groups = sorted(source_counts, key=lambda key: -source_counts[key])
    ax.barh(groups, [source_counts[g] for g in groups], color=[BLUE, GREEN, ORANGE][:len(groups)])
    for i, name in enumerate(groups):
        ax.text(source_counts[name] + 10, i, f"{source_counts[name]:,}  ({source_counts[name] / len(registry):.1%})", va="center")
    ax.set_xlim(0, max(source_counts.values()) * 1.3); ax.invert_yaxis()
    ax.set_xlabel("Lead records • source tags in the final registry")
    ax.set_title("B  Which routes supplied the lead pool?", loc="left", pad=13)

    ax = axes[1, 0]
    decision_labels = {"defer": "Defer • evidence unresolved", "investigate": "Investigate • retained hypothesis", "prioritize": "Prioritize • leading hypotheses", "exclude": "Exclude • not retained"}
    keys = [k for k in decision_labels if decisions[k]]
    bars = ax.barh([decision_labels[k] for k in keys], [decisions[k] for k in keys], color=[ORANGE, BLUE, GREEN, RED][:len(keys)])
    for bar, key in zip(bars, keys): ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f"{decisions[key]} / {len(assessed)}", va="center")
    ax.invert_yaxis(); ax.set_xlim(0, max(decisions.values()) * 1.28)
    ax.set_xlabel(f"Target-assessed records; {len(registry) - len(assessed):,} other leads have no target assessment")
    ax.set_title(f"C  What did the {len(assessed)} target reviews conclude?", loc="left", pad=13)

    ax = axes[1, 1]
    represented = [x for x in families if x["ranked_candidates"]]
    labels = [textwrap.fill(x["family_name"], 39) for x in represented]
    ax.barh(range(len(represented)), [x["ranked_candidates"] for x in represented], color=BLUE)
    for i, row in enumerate(represented): ax.text(row["ranked_candidates"] + .35, i, str(row["ranked_candidates"]), va="center", fontsize=9)
    ax.set_yticks(range(len(represented)), labels, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0, max(x["ranked_candidates"] for x in represented) * 1.16)
    ax.set_xlabel(f"Ranked accessions • {len(represented)} of {len(families)} registry family IDs represented")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("D  Which families reached the candidate portfolio?", loc="left", pad=13)

    ax = axes[2, 0]
    colors = {"prioritized hypothesis": GREEN, "candidate to investigate": BLUE, "deprioritized analogue": GRAY}
    pair_groups = defaultdict(list)
    for row in numeric_pairs:
        pair_groups[(row["partner_1_fit"], row["partner_2_fit"], row["tier"])].append(row)
    for tier, color in colors.items():
        grouped = [(key, rows) for key, rows in pair_groups.items() if key[2] == tier]
        if grouped:
            ax.scatter([k[0] for k, rows in grouped], [k[1] for k, rows in grouped],
                       s=[40 * len(rows) for k, rows in grouped], color=color, edgecolors="white", alpha=.8,
                       label=f"{tier} ({sum(len(rows) for k, rows in grouped)})")
    for key, rows in pair_groups.items():
        if key[2] == "prioritized hypothesis":
            name = "; ".join(dict.fromkeys(row["protein_name"] for row in rows))
            accessions = " · ".join(row["accession"] for row in rows)
            ax.annotate(textwrap.fill(name, 26) + "\n" + accessions, xy=(key[0], key[1]), xytext=(.04, .87),
                        textcoords="axes fraction", fontsize=8, color=GREEN,
                        arrowprops={"arrowstyle": "-", "color": GREEN, "lw": .8}, va="top")
    target_names = [f"Target partner {i + 1}" for i in range(2)]
    first_line = manifest["query"].splitlines()[0]
    if "ligate " in first_line and " and " in first_line.split("ligate ", 1)[1]:
        target_names = first_line.split("ligate ", 1)[1].rsplit(" and ", 1)
    ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel=f"{target_names[0]} analogue resemblance (0–1)", ylabel=f"{target_names[1]} analogue resemblance (0–1)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("E  Does the native reaction resemble both partners?", loc="left", pad=13)
    ax.text(.03, .03, f"{len(numeric_pairs)} of {len(ranked)} ranked candidates have two numeric comparisons\nArea = number sharing a point; resemblance is not activity", transform=ax.transAxes, fontsize=8.5, va="bottom")
    ax = axes[2, 1]
    fractions = [("Target assessments", len(assessed), len(registry)),
                 ("Two numeric substrate comparisons", len(numeric_pairs), len(ranked)),
                 ("Full-text downloads acquired", download_counts["acquired"], len(reading["downloads"]))]
    ax.barh(range(len(fractions)), [100] * len(fractions), color="#EDF1F5")
    ax.barh(range(len(fractions)), [100 * n/d for label, n, d in fractions], color=[GREEN, BLUE, ORANGE])
    for i, (label, n, d) in enumerate(fractions): ax.text(102, i, f"{n:,} / {d:,}\n{n/d:.0%}", va="center", fontsize=10)
    ax.set_yticks(range(len(fractions)), [textwrap.fill(x[0], 26) for x in fractions]); ax.invert_yaxis(); ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Different denominators are shown explicitly")
    ax.set_title("F  Where are the main evidence gaps?", loc="left", pad=13)
    save(fig, out, "overview")

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [1, 1.8, .8]})
    progression(axes[0]); axes[0].set_xlabel(""); axes[0].set_title("A  Lead accumulation and first target reviews", loc="left")
    status_colors = {"success": BLUE, "partial": ORANGE, "error": RED, "skipped": GRAY}
    for status, color in status_colors.items():
        records = [e for e in activity if e["status"] == status]
        axes[1].scatter([e["elapsed_minutes"] for e in records], [tool_names.index(e["tool"]) for e in records],
                        color=color, s=42, marker="x" if status == "error" else "o", label=f"{status} ({len(records)})")
    axes[1].set_yticks(range(len(tool_names)), [name.replace("_", " ") for name in tool_names], fontsize=9)
    axes[1].invert_yaxis(); axes[1].legend(loc="upper right", fontsize=9); axes[1].set_title("B  Which tools were used, and when? Each dot is one recorded tool call", loc="left")
    axes[2].step([0] + [r["elapsed_minutes"] for r in call_rows] + [end_minutes],
                 [0] + [r["cumulative_tokens"] / 1e6 for r in call_rows] + [call_rows[-1]["cumulative_tokens"] / 1e6], where="post", color=BLUE)
    axes[2].axhline(checkpoint["token_budget"] / 1e6, ls="--", color=ORANGE, label="Token budget")
    axes[2].set(xlabel="Minutes from discovery start", ylabel="Tokens (millions)")
    axes[2].legend(loc="upper left"); axes[2].set_title("C  Token use continued after the major lead-expansion burst", loc="left")
    fig.suptitle("Search timeline • timestamps from the event log, registry and review journal", fontsize=16)
    save(fig, out, "search_timeline")

    fig, axes = plt.subplots(1, 2, figsize=(18, 14), sharey=True, layout="constrained")
    colors = [GRAY, ORANGE, RED, BLUE]
    names = ["No target assessment", "Deferred", "Excluded", "Exported candidate"]
    for ax, scale in zip(axes, ["count", "fraction"]):
        left = [0.] * len(family_outcomes)
        totals = [sum(row[k] for k in outcome_keys) for row in family_outcomes]
        for key, color, label in zip(outcome_keys, colors, names):
            values = [row[key] if scale == "count" else 100 * row[key]/total for row, total in zip(family_outcomes, totals)]
            ax.barh(range(len(family_outcomes)), values, left=left, color=color, label=label)
            left = [a+b for a,b in zip(left, values)]
        ax.set_xlabel("Lead records (linear scale)" if scale == "count" else "Share of each family's recorded leads (%)")
        ax.set_title("Pool size" if scale == "count" else "Review coverage and outcome")
        if scale == "fraction":
            ax.set_xlim(0, 103)
        ax.legend(loc="lower right", fontsize=8)
    labels = [row["family_id"] + "  " + textwrap.fill(row["family_name"], 43)
              + f"  (n={sum(row[k] for k in outcome_keys)})" for row in family_outcomes]
    axes[0].set_yticks(range(len(family_outcomes)), labels, fontsize=8.5); axes[0].invert_yaxis()
    fig.suptitle(f"Family review coverage • all {len(registry):,} registry records, including missing family assignments\nCurrent review outcomes; family annotations and screening IDs are retained as recorded", fontsize=14)
    save(fig, out, "family_review_coverage")
    report = {"target_assessed_records": len(assessed), "no_target_assessment": len(registry) - len(assessed),
              "target_review_decisions": dict(decisions), "recorded_lead_source_groups": dict(source_counts),
              "numeric_partner_comparisons": len(numeric_pairs), "missing_numeric_partner_comparisons": len(ranked) - len(numeric_pairs),
              "fulltext_download_identifiers": len(reading["downloads"]), "fulltext_download_statuses": dict(download_counts),
              "paper_records": checkpoint["literature_coverage"]["paper_records"], "tool_calls": len(tool_events),
              "tool_status_counts": dict(Counter(e["status"] for e in tool_events)), "tool_types": len(tool_names),
              "research_decisions": len(research), "unassigned_family_records": sum(family_index["Unassigned"][k] for k in outcome_keys),
              "checks": {"exported_accessions_equal_prioritize_or_investigate_reviews": exported_ids == export_decisions,
                         "first_review_timestamps_cover_assessed_records": set(first_reviews) == assessed_ids,
                         "source_groups_partition_registry": sum(source_counts.values()) == len(registry),
                         "family_outcomes_partition_registry": sum(sum(row[k] for k in outcome_keys) for row in family_outcomes) == len(registry)}}
    (out / "search_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, research, activity
