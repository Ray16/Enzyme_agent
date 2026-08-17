#!/usr/bin/env python3
"""Render the enzyme_discovery_agent workflow pipeline as a simple diagram.

Generic and answer-agnostic (firewall-safe): it describes the pipeline's five
stages, never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_pipeline_figure.py
Outputs (next to this script): pipeline_workflow.png and pipeline_workflow.svg
"""
from pathlib import Path

from graphviz import Digraph

OUT = Path(__file__).resolve().parent / "pipeline_workflow"

# One box per stage: (id, title, subtitle, border, fill)
STAGES = [
    ("setup", "1 · Target setup",
     "resolve substrate → SMILES;\nderive bond type + direction", "#1f4e79", "#dbe9f6"),
    ("discovery", "2 · Discovery  (LLM proposes, tools find)",
     "literature + mechanism families →\nInterPro family survey →\nsubstrate-fit promotion of candidates",
     "#4b0082", "#e9d5ff"),
    ("evaluate", "3 · Screen + evaluate",
     "collapse orthologs; evaluate every\ngene representative (parallel, no LLM)",
     "#0b6b3a", "#d6f0e0"),
    ("rank", "4 · Rank",
     "deterministic mechanistic spine\n+ LLM focus-rank", "#8a1f1f", "#ffd9d9"),
    ("report", "5 · Report",
     "confidence + explorative reports,\nranked candidates, structures", "#333333", "#eeeeee"),
]


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline", format="png")
    g.attr(rankdir="TB", bgcolor="white", ranksep="0.5")
    g.attr("edge", color="#666666", penwidth="1.5", arrowsize="0.9")

    for sid, title, sub, fg, bg in STAGES:
        label = (f'<<b>{title}</b><br/>'
                 f'<font point-size="10">{sub.replace(chr(10), "<br/>")}</font>>')
        g.node(sid, label, shape="box", style="filled,rounded",
               fillcolor=bg, color=fg, fontcolor=fg, penwidth="2",
               fontname="Helvetica", fontsize="13", margin="0.25,0.16")

    for (a, *_), (b, *_) in zip(STAGES, STAGES[1:]):
        g.edge(a, b)

    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    g.render(str(OUT), format="svg", cleanup=True)
    print(f"wrote {OUT}.png and {OUT}.svg")


if __name__ == "__main__":
    main()
