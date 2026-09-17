#!/usr/bin/env python3
"""Render the enzyme_discovery_agent pipeline as a top-down TREE: a root that
branches into the three phases, each phase a vertical chain of its ordered
steps, and the Rank step expanding into the deterministic P0->P6 priority spine.

Complements ``generate_pipeline_figure.py`` (the horizontal 5-card summary) with
a hierarchical view that shows the sub-steps inside each phase.

Generic and answer-agnostic (firewall-safe): describes the pipeline only; never
a specific target, substrate, or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_pipeline_tree.py
Outputs (next to this script): pipeline_tree.png (white background).
"""
from pathlib import Path

from graphviz import Digraph

OUT = Path(__file__).resolve().parent / "pipeline_tree"

FONT = "Lato"
INK = "#0f172a"
MUTED = "#475569"
LINE = "#b8c2d0"

# (phase_id, number, title, accent, [ (step_id, title, descriptor) ... ] )
PHASES = [
    ("p1", "1", "Reaction analysis", "#2563eb", [
        ("decompose", "Decompose reaction", "bond type + direction (formation vs. hydrolysis)"),
        ("mechanisms", "Enumerate mechanisms", "2-4 distinct folds, each a different intermediate"),
    ]),
    ("p2", "2", "Discovery", "#7c3aed", [
        ("lit", "Literature harvest", "early read seeds which families to explore"),
        ("search", "Broad database search", "seed the pool across all mechanisms"),
        ("family", "Per-family depth", "resolve members, pull reps, expand orthologs"),
        ("screen", "Screen candidates", "reject wrong direction / wrong reaction"),
        ("coverage", "Coverage gap-check", "every family gets a screened representative"),
    ]),
    ("p3", "3", "Evaluate &amp; rank", "#0d9488", [
        ("evaluate", "Full-text + evaluate", "mine mechanism + measurements, score fit"),
        ("dedup", "Ortholog dedup", "collapse to distinct enzymes before ranking"),
        ("rank", "Rank (lexicographic spine)", "deterministic priority over grounded facts"),
        ("tier", "Tier", "T1 strong lead / T2 worth testing / T3 weak-gated"),
        ("report", "Report + handoff", "CSV, reports, structural-modeling handoff"),
    ]),
]

# The deterministic ranking spine expanded under the Rank step (P0 -> P6).
SPINE = [
    ("P0", "hard gate", "critical warning &#8594; bottom (Tier 3)"),
    ("PF", "fold decision", "on-target &gt; neutral &gt; stopped"),
    ("P1", "substrate", "functional-group similarity"),
    ("P2", "mechanism", "grounded &gt; supported &gt; tentative"),
    ("P3", "curation", "Swiss-Prot reviewed higher"),
    ("P4", "depth", "characterization papers"),
    ("P5", "structure", "PDB &gt; AlphaFold &gt; none"),
    ("P6", "engineering", "engineering precedent"),
]


def _tint(hex_color: str, amount: float) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _root_card() -> str:
    return (
        '<<table border="0" cellborder="0" cellspacing="0" cellpadding="12" '
        'style="rounded" bgcolor="#0f172a"><tr><td>'
        '<font point-size="18" color="white"><b>Enzyme Discovery Agent</b></font><br/>'
        '<font point-size="11" color="#cbd5e1">LLM proposes strategies &#183; tools find entities '
        '(data-leakage firewall)</font>'
        '</td></tr></table>>'
    )


def _phase_card(num, title, accent) -> str:
    return (
        f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="10" '
        f'style="rounded" bgcolor="{accent}"><tr><td align="center">'
        f'<font point-size="15" color="white"><b>Phase {num}</b></font><br/>'
        f'<font point-size="13" color="white">{title}</font>'
        f'</td></tr></table>>'
    )


def _step_card(title, desc, accent) -> str:
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="8" style="rounded" bgcolor="{_tint(accent, 0.94)}" width="240">'
        f'<tr><td align="left" width="240">'
        f'<font point-size="12.5" color="{INK}"><b>{title}</b></font><br/>'
        f'<font point-size="10" color="{MUTED}">{desc}</font>'
        f'</td></tr></table>>'
    )


def _spine_card(accent) -> str:
    rows = "".join(
        f'<tr><td align="right"><font point-size="10.5" color="{accent}"><b>{tag}</b></font></td>'
        f'<td width="8"></td>'
        f'<td align="left"><font point-size="10.5" color="{INK}"><b>{name}</b></font> '
        f'<font point-size="10" color="{MUTED}">&#183; {desc}</font></td></tr>'
        for tag, name, desc in SPINE
    )
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="4" style="rounded" bgcolor="white">'
        f'<tr><td colspan="3" bgcolor="{accent}" cellpadding="6" align="center">'
        f'<font point-size="11.5" color="white"><b>Ranking spine &#183; strict lexicographic order</b></font>'
        f'</td></tr>{rows}</table>>'
    )


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline_tree")
    g.attr(rankdir="TB", bgcolor="white", dpi="220", pad="0.4",
           ranksep="0.5", nodesep="0.4", fontname=FONT)
    g.attr("node", fontname=FONT, shape="plain")
    g.attr("edge", color=LINE, penwidth="2.6", arrowsize="1.0")

    g.node("root", _root_card())

    for pid, num, title, accent, steps in PHASES:
        g.node(pid, _phase_card(num, title, accent))
        g.edge("root", pid)
        prev = pid
        for sid, stitle, sdesc in steps:
            g.node(sid, _step_card(stitle, sdesc, accent))
            g.edge(prev, sid)
            prev = sid
            if sid == "rank":                       # expand the spine under Rank
                g.node("spine", _spine_card(accent))
                g.edge("rank", "spine", style="dashed", penwidth="2.0",
                       color=accent, constraint="false")

    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
