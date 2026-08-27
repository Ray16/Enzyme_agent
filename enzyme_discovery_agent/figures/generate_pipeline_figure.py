#!/usr/bin/env python3
"""Render the enzyme_discovery_agent workflow pipeline as a clean, slide-ready
figure: five balanced stage cards, left → right, with the two literature
evidence-extractor tools shown feeding the stages they inform.

Generic and answer-agnostic (firewall-safe): it describes the pipeline's stages,
never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_pipeline_figure.py
Outputs (next to this script): pipeline_workflow.png (white background).
"""
from pathlib import Path

from graphviz import Digraph

OUT = Path(__file__).resolve().parent / "pipeline_workflow"

FONT = "Lato"
INK = "#0f172a"
MUTED = "#475569"
FAINT = "#94a3b8"
LINE = "#b8c2d0"

# One card per stage: (id, number, title, two short descriptor lines, accent)
STAGES = [
    ("setup",     "1", "Target setup", ["read the substrates,", "work out the reaction"], "#2563eb"),
    ("discovery", "2", "Discovery",    ["literature + chemistry", "+ family search"],     "#7c3aed"),
    ("evaluate",  "3", "Evaluate",     ["check every candidate", "in parallel"],          "#0d9488"),
    ("rank",      "4", "Rank",         ["by evidence,", "then a model re-rank"],           "#e11d48"),
    ("report",    "5", "Report",       ["ranked enzymes,", "reports &amp; structures"],    "#334155"),
]

# Literature evidence-extractor tools, drawn one rank below and anchored under
# the stage they primarily feed (mechanism -> Evaluate/Rank; measurements ->
# Report), so the feed arrows stay short and do not cross.
#   (id, name, desc, accent, anchor_stage, [stages_fed])
EXTRACTORS = [
    ("mech", "extract_reaction_mechanism", "grounded mechanism facts",
     "#7c3aed", "evaluate", ["evaluate", "rank"]),
    ("meas", "extract_experimental_measurements", "kinetics / activity rows",
     "#0d9488", "report", ["report"]),
]


def _tint(hex_color: str, amount: float = 0.90) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _stage_card(num, title, desc, accent) -> str:
    d0, d1 = desc
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="0" style="rounded" bgcolor="white" width="196">'
        f'<tr><td bgcolor="{accent}" cellpadding="9" align="center">'
        f'<font point-size="16" color="white"><b>{num}&#160;&#160;&#160;{title}</b></font></td></tr>'
        f'<tr><td cellpadding="14" align="center">'
        f'<font point-size="12" color="{MUTED}">{d0}<br/>{d1}</font></td></tr>'
        f'</table>>'
    )


def _extractor_card(name, desc, accent) -> str:
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="0" style="rounded" bgcolor="{_tint(accent, 0.96)}">'
        f'<tr><td bgcolor="{accent}" cellpadding="5">'
        f'<font point-size="9.5" color="white"><b>EVIDENCE TOOL</b></font></td></tr>'
        f'<tr><td cellpadding="6"><font point-size="13" color="{INK}"><b>{name}</b></font>'
        f'<br/><font point-size="10.5" color="{MUTED}">{desc}</font></td></tr>'
        f'<tr><td height="4"></td></tr>'
        f'</table>>'
    )


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline")
    g.attr(rankdir="LR", bgcolor="white", dpi="220", pad="0.35",
           ranksep="0.7", nodesep="0.45", fontname=FONT)
    g.attr("node", fontname=FONT, shape="plain")

    # Main stage row.
    for sid, num, title, desc, accent in STAGES:
        g.node(sid, _stage_card(num, title, desc, accent))
    for (a, *_), (b, *_) in zip(STAGES, STAGES[1:]):
        g.edge(a, b, color=LINE, penwidth="2.8", arrowsize="1.15")

    # Evidence-extractor tools: place each in the SAME rank (column, under LR) as
    # its anchor stage, so it sits directly below that stage and the feed arrows
    # stay short and non-crossing.
    for eid, name, desc, accent, anchor, _feeds in EXTRACTORS:
        with g.subgraph() as s:
            s.attr(rank="same")
            s.node(anchor)                                   # reference existing stage
            s.node(eid, _extractor_card(name, desc, accent))

    for eid, _name, _desc, accent, _anchor, feeds in EXTRACTORS:
        for stage in feeds:
            g.edge(eid, stage, color=accent, penwidth="2.2", arrowsize="0.9",
                   style="dashed", constraint="false")

    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
