#!/usr/bin/env python3
"""Render the enzyme_discovery_agent workflow pipeline as a clean, slide-ready
figure: five balanced cards, left → right, high resolution.

Generic and answer-agnostic (firewall-safe): it describes the pipeline's stages,
never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_pipeline_figure.py
Outputs (next to this script): pipeline_workflow.png and pipeline_workflow.svg
"""
from pathlib import Path

from graphviz import Digraph

OUT = Path(__file__).resolve().parent / "pipeline_workflow"

# One card per stage: (id, number, title, two short descriptor lines, accent)
STAGES = [
    ("setup",     "1", "Target setup", ["read the substrates,", "work out the reaction"],      "#2563eb"),
    ("discovery", "2", "Discovery",     ["literature + chemistry", "+ family search"],          "#7c3aed"),
    ("evaluate",  "3", "Evaluate",      ["check every candidate", "in parallel"],               "#0d9488"),
    ("rank",      "4", "Rank",          ["by evidence,", "then a model re-rank"],                "#e11d48"),
    ("report",    "5", "Report",        ["ranked enzymes,", "reports &amp; structures"],         "#334155"),
]


def _tint(hex_color: str, amount: float = 0.90) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


# Literature evidence-extractor tools, shown as a band feeding the stages they
# inform: mechanism facts drive the Evaluate/Rank feasibility judgement, and
# measured kinetics/activity are gathered for the candidates being modelled.
EXTRACTORS = [
    ("mech", "extract_reaction_mechanism",
     "grounded mechanism facts", "#7c3aed", ["evaluate", "rank"]),
    ("meas", "extract_experimental_measurements",
     "kinetics / activity rows", "#0d9488", ["report"]),
]


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline")
    g.attr(rankdir="LR", bgcolor="white", dpi="220", ranksep="0.55",
           fontname="Helvetica")
    g.attr("node", fontname="Helvetica", shape="box", style="filled,rounded",
           fixedsize="true", width="2.15", height="1.65", margin="0")

    for sid, num, title, desc, accent in STAGES:
        d0, d1 = desc
        label = (
            f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="0">'
            f'<tr><td height="6"></td></tr>'
            f'<tr><td><font point-size="34" color="{accent}"><b>{num}</b></font></td></tr>'
            f'<tr><td><font point-size="17" color="#0f172a"><b>{title}</b></font></td></tr>'
            f'<tr><td height="6"></td></tr>'
            f'<tr><td><font point-size="11.5" color="#475569">{d0}<br/>{d1}</font></td></tr>'
            f'</table>>'
        )
        g.node(sid, label, fillcolor=_tint(accent), color=accent, penwidth="2.4")

    for (a, *_), (b, *_) in zip(STAGES, STAGES[1:]):
        g.edge(a, b, color="#94a3b8", penwidth="2.8", arrowsize="1.15")

    # Evidence-extractor band: keep the extractor nodes on one rank below the
    # main flow, then connect each up (constraint-free) to the stage it feeds.
    with g.subgraph() as row:
        row.attr(rank="same")
        for eid, name, desc, accent, _feeds in EXTRACTORS:
            label = (
                f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="0">'
                f'<tr><td><font point-size="9.5" color="{accent}"><b>EVIDENCE TOOL</b></font></td></tr>'
                f'<tr><td><font point-size="12.5" color="#0f172a"><b>{name}</b></font></td></tr>'
                f'<tr><td height="3"></td></tr>'
                f'<tr><td><font point-size="10.5" color="#475569">{desc}</font></td></tr>'
                f'</table>>'
            )
            row.node(eid, label, fillcolor=_tint(accent, 0.94), color=accent,
                     penwidth="2.0", width="4.1", height="1.05",
                     style="filled,rounded,dashed")

    for eid, _name, _desc, accent, feeds in EXTRACTORS:
        for stage in feeds:
            g.edge(eid, stage, color=accent, penwidth="2.0", arrowsize="0.9",
                   style="dashed", constraint="false")

    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    g.render(str(OUT), format="svg", cleanup=True)
    print(f"wrote {OUT}.png and {OUT}.svg")


if __name__ == "__main__":
    main()
