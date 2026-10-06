#!/usr/bin/env python3
"""Render the enzyme_discovery_agent pipeline as a REASONING-CENTERED figure.

The agent's core capability is reasoning; tools are the instruments it uses to
test each hypothesis. So each stage card leads with the reasoning act (header)
and the decision the agent makes (body), and lists the tools it draws on only as
a small muted footer. A single strip below names the reasoning safeguards.

Generic and answer-agnostic (firewall-safe): describes the pipeline's reasoning,
never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_pipeline_reasoning.py
Output (next to this script): pipeline_reasoning.png (white background).
"""
from pathlib import Path

from graphviz import Digraph

OUT = Path(__file__).resolve().parent / "pipeline_reasoning"

FONT = "Lato"
INK = "#0f172a"
MUTED = "#475569"
FAINT = "#94a3b8"
LINE = "#b8c2d0"

# (id, number, reasoning-act title, body lines [the DECISION], tool chip text, accent)
STAGES = [
    ("ra", "1", "Analyze the reaction",
     ["Decompose the target chemistry;",
      "<b>deduce candidate fold families</b>"],
     "chemistry (RDKit)", "#2563eb"),
    ("disc", "2", "Pursue the hypotheses",
     ["Search per family, keep real leads;",
      "screen a <b>family-diverse</b> set"],
     "literature &#183; UniProt / KEGG / InterPro", "#7c3aed"),
    ("eval", "3", "Judge compatibility",
     ["Right direction, bond &amp; substrate fit",
      "from decomposed facts, not one score"],
     "annotations &#183; EC bond-type &#183; structures", "#0d9488"),
    ("rank", "4", "Rank with reasons",
     ["Weigh the evidence into a",
      "<b>justified, inspectable shortlist</b>"],
     "evidence weighting", "#e11d48"),
]


def _tint(hex_color: str, amount: float = 0.90) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _wrap(inner, width) -> str:
    return (f'<table border="0" cellborder="0" cellspacing="0" cellpadding="0" '
            f'width="{width}"><tr><td align="center">{inner}</td></tr></table>')


def _stage_card(num, title, body, accent) -> str:
    header = f'<font point-size="16" color="white"><b>{num}&#160;&#160;{title}</b></font>'
    body_html = "<br/>".join(body)
    body_font = f'<font point-size="12" color="{INK}">{body_html}</font>'
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="0" style="rounded" bgcolor="white">'
        f'<tr><td bgcolor="{accent}" cellpadding="10">{_wrap(header, 210)}</td></tr>'
        f'<tr><td cellpadding="13">{_wrap(body_font, 210)}</td></tr>'
        f'</table>>'
    )


def _tool_chip(text) -> str:
    """A small, muted chip naming the tools a step invokes -- deliberately minor,
    so the reasoning card above it reads as the substance."""
    return (
        f'<<table border="1" color="{LINE}" cellborder="0" cellspacing="0" '
        f'cellpadding="7" style="rounded" bgcolor="#f8fafc">'
        f'<tr><td>{_wrap(f"<font point-size=\'8.5\' color=\'{FAINT}\'><b>TOOLS</b></font>", 150)}</td></tr>'
        f'<tr><td>{_wrap(f"<font point-size=\'9.5\' color=\'{MUTED}\'>{text}</font>", 150)}</td></tr>'
        f'</table>>'
    )


def _pill(text, accent) -> str:
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="10" style="rounded" bgcolor="{_tint(accent, 0.92)}">'
        f'<tr><td>{_wrap(f"<font point-size=\'12\' color=\'{INK}\'><b>{text}</b></font>", 120)}'
        f'</td></tr></table>>'
    )


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline_reasoning")
    # Top-to-bottom vertical spine: compact for a proposal page (portrait).
    g.attr(rankdir="TB", bgcolor="white", dpi="220", pad="0.3",
           ranksep="0.42", nodesep="0.55", fontname=FONT)
    g.attr("node", fontname=FONT, shape="plain")

    g.node("in", _pill("Target reaction", "#334155"))
    for sid, num, title, body, tool, accent in STAGES:
        g.node(sid, _stage_card(num, title, body, accent))
        g.node(sid + "_t", _tool_chip(tool))
    g.node("out", _pill("Ranked candidates", "#334155"))

    # Reasoning spine (bold vertical flow).
    order = ["in"] + [s[0] for s in STAGES] + ["out"]
    for a, b in zip(order, order[1:]):
        g.edge(a, b, color=LINE, penwidth="2.8", arrowsize="1.0")

    # Tool chips hang to the RIGHT of each reasoning step, as subordinate notes.
    for sid, *_ in STAGES:
        with g.subgraph() as s:
            s.attr(rank="same")
            s.node(sid)
            s.node(sid + "_t")
        g.edge(sid, sid + "_t", color=FAINT, penwidth="1.2", style="dashed",
               dir="none", constraint="false")
    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
