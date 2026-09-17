#!/usr/bin/env python3
"""Render the enzyme_discovery_agent pipeline as a top-down TREE: a root that
branches into the three phases, each phase a vertical chain of its ordered
steps, and the Rank step expanding into the deterministic priority spine.

Complements ``generate_pipeline_figure.py`` (the horizontal 5-card summary) with
a hierarchical view that shows the sub-steps inside each phase.  Every label is
written to be self-explanatory at a glance (no internal jargon).

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
# Descriptors may use <br/> to wrap onto a second line.
PHASES = [
    ("p1", "1", "Reaction analysis", "#2563eb", [
        ("decompose", "Understand the reaction",
         "Which chemical bond is made or broken,<br/>and in which direction (build vs. break)"),
        ("mechanisms", "List possible mechanisms",
         "2-4 distinct catalytic routes, each using<br/>a different protein fold &amp; intermediate"),
    ]),
    ("p2", "2", "Enzyme discovery", "#7c3aed", [
        ("lit", "Read the literature",
         "Learn which enzyme families<br/>are worth exploring"),
        ("search", "Search databases broadly",
         "Collect candidate enzymes from<br/>every proposed mechanism"),
        ("family", "Go deep in each family",
         "Pull the real family members<br/>and their close relatives"),
        ("screen", "Screen out mismatches",
         "Drop enzymes that run the<br/>wrong reaction or wrong direction"),
        ("coverage", "Check for coverage gaps",
         "Make sure every family has at<br/>least one candidate examined"),
    ]),
    ("p3", "3", "Evaluation &amp; ranking", "#0d9488", [
        ("evaluate", "Evaluate each candidate",
         "Read full papers for mechanism &amp; activity,<br/>then score how well the enzyme fits"),
        ("dedup", "Remove duplicates",
         "Merge near-identical enzymes so<br/>only distinct ones are ranked"),
        ("rank", "Rank by priority rules",
         "Order candidates using fixed rules<br/>over verified facts (see below)"),
        ("tier", "Sort into confidence tiers",
         "Tier 1 strong lead &#183; Tier 2 worth<br/>testing &#183; Tier 3 weak"),
        ("report", "Report &amp; hand off",
         "Ranked table + written reports;<br/>pass leads to structure modeling"),
    ]),
]

# The ranking priority rules expanded under the Rank step. Applied strictly in
# order: each rule only breaks ties left by the rule above it.
SPINE = [
    ("1", "Safety gate", "any critical flaw sinks the candidate to the bottom"),
    ("2", "Fold match", "correct fold beats an unrelated one beats a known-wrong one"),
    ("3", "Substrate fit", "closer functional-group match ranks higher"),
    ("4", "Mechanism evidence", "experimentally proven &gt; supported &gt; only hinted"),
    ("5", "Curation", "manually reviewed (Swiss-Prot) entries rank higher"),
    ("6", "Study depth", "more characterization papers ranks higher"),
    ("7", "Structure", "experimental (PDB) &gt; predicted (AlphaFold) &gt; none"),
    ("8", "Engineering record", "prior successful protein engineering ranks higher"),
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
        '<font point-size="11" color="#cbd5e1">The AI plans the search &#183; tools retrieve the real '
        'enzymes<br/>&#8212; the answer is never shown to it (data-leakage firewall)</font>'
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
        f'cellpadding="9" style="rounded" bgcolor="{_tint(accent, 0.94)}" width="270">'
        f'<tr><td align="left" width="270">'
        f'<font point-size="13" color="{INK}"><b>{title}</b></font><br/>'
        f'<font point-size="10.5" color="{MUTED}">{desc}</font>'
        f'</td></tr></table>>'
    )


def _spine_card(accent) -> str:
    rows = "".join(
        f'<tr><td align="right"><font point-size="11" color="{accent}"><b>{tag}</b></font></td>'
        f'<td width="10"></td>'
        f'<td align="left"><font point-size="11" color="{INK}"><b>{name}</b></font> '
        f'<font point-size="10.5" color="{MUTED}">&#183; {desc}</font></td></tr>'
        for tag, name, desc in SPINE
    )
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="5" style="rounded" bgcolor="white">'
        f'<tr><td colspan="3" bgcolor="{accent}" cellpadding="7" align="center">'
        f'<font point-size="12" color="white"><b>Ranking rules &#8212; applied in order; '
        f'each only breaks ties left by the rule above</b></font>'
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
            if sid == "rank":                       # spine sits in-flow, below Rank
                g.node("spine", _spine_card(accent))
                g.edge("rank", "spine")
                prev = "spine"

    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
