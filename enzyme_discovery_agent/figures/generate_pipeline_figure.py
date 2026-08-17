#!/usr/bin/env python3
"""Render the enzyme_discovery_agent workflow pipeline as a clean, slide-ready
figure: five stages, left → right, high resolution.

Generic and answer-agnostic (firewall-safe): it describes the pipeline's stages,
never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_pipeline_figure.py
Outputs (next to this script): pipeline_workflow.png and pipeline_workflow.svg
"""
from pathlib import Path

from graphviz import Digraph

OUT = Path(__file__).resolve().parent / "pipeline_workflow"

# One card per stage: (id, number, title, one-line descriptor, accent colour)
STAGES = [
    ("setup",     "1", "Target setup",     "resolve substrate → SMILES;\nderive bond type &amp; direction",           "#2563eb"),
    ("discovery", "2", "Discovery",         "literature + mechanism families →\nInterPro survey → promote candidates", "#7c3aed"),
    ("evaluate",  "3", "Evaluate",          "one representative per gene;\nevaluate every one (parallel)",             "#0d9488"),
    ("rank",      "4", "Rank",              "deterministic spine\n+ LLM focus-rank",                                   "#e11d48"),
    ("report",    "5", "Report",            "confidence &amp; explorative reports,\nranked candidates, structures",     "#334155"),
]


def _tint(hex_color: str) -> str:
    """A very light tint of the accent for the card fill."""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * 0.88)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline")
    g.attr(rankdir="LR", bgcolor="white", dpi="220", ranksep="0.7",
           fontname="Helvetica", label=(
               "\nEnzyme Discovery Agent — workflow\n"
               "LLM proposes strategy · deterministic gates + tools find the enzymes"),
           labelloc="t", fontsize="20", fontcolor="#0f172a")
    g.attr("node", fontname="Helvetica")

    for sid, num, title, desc, accent in STAGES:
        fill = _tint(accent)
        label = (
            f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="4">'
            f'<tr><td><font point-size="30" color="{accent}"><b>{num}</b></font>'
            f'&nbsp;&nbsp;<font point-size="19" color="#0f172a"><b>{title}</b></font></td></tr>'
            f'<tr><td><font point-size="13" color="#334155">'
            f'{desc.replace(chr(10), "<br/>")}</font></td></tr>'
            f'</table>>'
        )
        g.node(sid, label, shape="box", style="filled,rounded",
               fillcolor=fill, color=accent, penwidth="2.4", margin="0.28,0.20")

    for (a, *_), (b, *_) in zip(STAGES, STAGES[1:]):
        g.edge(a, b, color="#94a3b8", penwidth="2.6", arrowsize="1.1")

    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    g.render(str(OUT), format="svg", cleanup=True)
    print(f"wrote {OUT}.png and {OUT}.svg")


if __name__ == "__main__":
    main()
