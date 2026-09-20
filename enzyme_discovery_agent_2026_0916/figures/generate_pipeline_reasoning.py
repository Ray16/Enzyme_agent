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

# (id, number, reasoning-act title, body lines [the DECISION], tool footer, accent)
STAGES = [
    ("ra", "1", "Analyze the reaction",
     ["Decompose bond type, direction, functional",
      "groups &amp; activation chemistry, then",
      "<b>deduce candidate fold families</b>"],
     "chemistry reasoning", "#2563eb"),
    ("disc", "2", "Pursue the hypotheses",
     ["Search literature &amp; databases per family,",
      "keep genuine leads over keyword matches,",
      "screen a <b>family-diverse</b> candidate set"],
     "federated literature &#183; UniProt / KEGG / InterPro &#183; citations", "#7c3aed"),
    ("eval", "3", "Judge compatibility",
     ["Per candidate, reason: right direction?",
      "right bond? substrate fit? &#8212; from",
      "decomposed facts, not one similarity score"],
     "EC bond-type &#183; catalytic annotations &#183; structures", "#0d9488"),
    ("rank", "4", "Rank with reasons",
     ["Weigh the evidence dimensions into a",
      "<b>justified shortlist</b> in which every",
      "placement is backed by explicit reasoning"],
     "deterministic priority over evidence", "#e11d48"),
]

SAFEGUARDS = ("Reasoning safeguards:  data-leakage firewall (families come from "
              "reasoning, not the answer)  &#183;  every proposed family must be "
              "covered before concluding  &#183;  multi-substrate fit scored by the "
              "worst-matched substrate")


def _tint(hex_color: str, amount: float = 0.90) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _wrap(inner, width) -> str:
    return (f'<table border="0" cellborder="0" cellspacing="0" cellpadding="0" '
            f'width="{width}"><tr><td align="center">{inner}</td></tr></table>')


def _stage_card(num, title, body, footer, accent) -> str:
    header = f'<font point-size="15" color="white"><b>{num}&#160;&#160;{title}</b></font>'
    body_html = "<br/>".join(body)
    body_font = f'<font point-size="11.5" color="{MUTED}">{body_html}</font>'
    foot = (f'<font point-size="9.5" color="{FAINT}">uses:&#160;{footer}</font>')
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="0" style="rounded" bgcolor="white">'
        f'<tr><td bgcolor="{accent}" cellpadding="9">{_wrap(header, 210)}</td></tr>'
        f'<tr><td cellpadding="12">{_wrap(body_font, 210)}</td></tr>'
        f'<tr><td cellpadding="7">{_wrap(foot, 210)}</td></tr>'
        f'</table>>'
    )


def _pill(text, accent) -> str:
    return (
        f'<<table border="2" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="10" style="rounded" bgcolor="{_tint(accent, 0.92)}">'
        f'<tr><td>{_wrap(f"<font point-size=\'12\' color=\'{INK}\'><b>{text}</b></font>", 120)}'
        f'</td></tr></table>>'
    )


def _note(text) -> str:
    return (f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="8">'
            f'<tr><td>{_wrap(f"<font point-size=\'10.5\' color=\'{MUTED}\'>{text}</font>", 1180)}'
            f'</td></tr></table>>')


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline_reasoning")
    g.attr(rankdir="LR", bgcolor="white", dpi="220", pad="0.35",
           ranksep="0.55", nodesep="0.45", fontname=FONT,
           label=('<<font point-size="20" color="#0f172a"><b>Reasoning-driven enzyme '
                  'discovery</b></font><br/><font point-size="11.5" color="#94a3b8">'
                  'the agent reasons about the chemistry, then uses tools to test each '
                  'hypothesis</font>>'),
           labelloc="t", labeljust="c")
    g.attr("node", fontname=FONT, shape="plain")

    g.node("in", _pill("Target<br/>reaction", "#334155"))
    for sid, num, title, body, footer, accent in STAGES:
        g.node(sid, _stage_card(num, title, body, footer, accent))
    g.node("out", _pill("Ranked<br/>candidates", "#334155"))

    order = ["in"] + [s[0] for s in STAGES] + ["out"]
    for a, b in zip(order, order[1:]):
        g.edge(a, b, color=LINE, penwidth="2.8", arrowsize="1.1")

    # Safeguards strip, spanning below the flow.
    g.node("safe", _note(SAFEGUARDS))
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("eval")
        s.node("safe")
    g.edge("safe", "eval", style="invis")
    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
