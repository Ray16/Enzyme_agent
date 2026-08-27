#!/usr/bin/env python3
"""Render internal-flow figures for the two literature evidence-extractor tools:

  * extract_reaction_mechanism        -> mechanism_extraction_flow.png
  * extract_experimental_measurements -> measurement_extraction_flow.png

Each figure shows how the tool works internally (discovery -> read -> extract ->
score/validate -> output), faithfully to the code in
``enzyme_discovery_agent/tools/{mechanism,measurement}_extraction.py``.

Generic and answer-agnostic (firewall-safe): the diagrams describe tool
mechanics only, never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_tool_figures.py
Outputs (next to this script): the two *_flow.png files (white background).
"""
from pathlib import Path

from graphviz import Digraph

HERE = Path(__file__).resolve().parent

FONT = "Lato"          # clean modern sans (falls back to the default sans if absent)
INK = "#0f172a"        # near-black title ink
MUTED = "#475569"      # body text
FAINT = "#94a3b8"      # de-emphasised / optional
LINE = "#b8c2d0"       # default edge


def _tint(hex_color: str, amount: float) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


# --- HTML-label building blocks ------------------------------------------- #
def _lines(items, size=10.5, color=MUTED) -> str:
    return f'<font point-size="{size}" color="{color}">{"<br/>".join(items)}</font>'


def _stack(*parts) -> str:
    rows = "".join(f'<tr><td align="center">{p}</td></tr>' for p in parts)
    return (f'<table border="0" cellborder="0" cellspacing="0" cellpadding="0">'
            f'{rows}</table>')


def _two_col(left, right, size=10.5, color=MUTED) -> str:
    l = f'<font point-size="{size}" color="{color}">{"<br/>".join(left)}</font>'
    r = f'<font point-size="{size}" color="{color}">{"<br/>".join(right)}</font>'
    return (f'<table border="0" cellborder="0" cellspacing="0" cellpadding="0">'
            f'<tr><td align="left" balign="left">{l}</td>'
            f'<td width="26"></td>'
            f'<td align="left" balign="left">{r}</td></tr></table>')


def _card(title, body_html, accent, title_size=14, header_fg="white", border="2") -> str:
    """A card node: solid accent header band (white title) over a white body."""
    return (
        f'<<table border="{border}" color="{accent}" cellborder="0" cellspacing="0" '
        f'cellpadding="0" style="rounded" bgcolor="white">'
        f'<tr><td bgcolor="{accent}" cellpadding="8" align="center">'
        f'<font point-size="{title_size}" color="{header_fg}"><b>{title}</b></font></td></tr>'
        f'<tr><td cellpadding="10" align="center">{body_html}</td></tr>'
        f'</table>>'
    )


def _base(name: str, title: str, subtitle: str) -> Digraph:
    g = Digraph(name)
    g.attr(rankdir="TB", bgcolor="white", dpi="220", pad="0.35",
           ranksep="0.55", nodesep="0.40", fontname=FONT,
           label=(f'<<font point-size="20" color="{INK}"><b>{title}</b></font>'
                  f'<br/><font point-size="11.5" color="{FAINT}">{subtitle}</font>'
                  f'<br/> >'),
           labelloc="t", labeljust="c")
    g.attr("node", shape="plain", fontname=FONT)
    g.attr("edge", color=LINE, penwidth="2.0", arrowsize="0.85", fontname=FONT)
    return g


def _cluster(c, label, accent):
    c.attr(label=f"  {label}", fontname=FONT, fontsize="11",
           color=_tint(accent, 0.45), style="rounded,filled",
           fillcolor=_tint(accent, 0.965), fontcolor=_tint(accent, 0.25),
           penwidth="1.4", margin="12")


# --------------------------------------------------------------------------- #
#  extract_reaction_mechanism                                                  #
# --------------------------------------------------------------------------- #
def build_mechanism() -> Digraph:
    A = "#7c3aed"   # purple — feeds Evaluate / Rank
    g = _base("mechanism_extraction_flow", "extract_reaction_mechanism",
              "grounded, reaction-class-agnostic mechanism facts from primary literature")

    g.node("in", _card("Input",
                       _lines(["enzyme name / EC / accession",
                               "optional: specific paper (DOI / PMID / PMCID)"]), A))

    with g.subgraph(name="cluster_disc") as c:
        _cluster(c, "candidate discovery  (skipped when a paper is supplied)", A)
        c.node("src", _card("Federated search &#183; 4 sources in parallel",
                            _lines(["PubMed &#183; OpenAlex &#183; Europe PMC &#183; Semantic Scholar",
                                    "class-agnostic query (no &#8216;nucleophile&#8217; bias)"]), A))
        c.node("merge", _card("consolidate_pool",
                             _lines(["cross-source dedup / merge (back-fills PMCID)",
                                     "semantic RRF rerank &#8594; ranked candidates"]), A))
        c.node("fallback", _card("fallback",
                                _lines(["Europe PMC keyword search",
                                        "+ mechanism / structure keyword ranking"], color=MUTED),
                                FAINT))
        c.edge("src", "merge")
        c.edge("src", "fallback", style="dashed", color=FAINT,
               label=" stack down ", fontsize="9", fontcolor=FAINT)

    g.node("loop", _card("For each candidate, best-first &#183; early-stop when confident",
                        _lines(["fetch full text:  JATS &#8594; text   or   PDF &#8594; text",
                                "LLM extract mechanism JSON (14k-char excerpt)",
                                "score groundedness &#8594; keep best"]), A))
    g.node("gate", _card("Groundedness gate &#183; _score_facts",
                        _lines(["named participants + bonds + residues / cofactors?",
                                "penalise &#8216;no detail&#8217; disclaimers &amp; bare-water Nu"]), A))

    out_body = _stack(
        _two_col(["mechanism_class &#183; chemical_step",
                  "bonds broken / formed",
                  "participants &#183; reactive_pair",
                  "catalytic_residues"],
                 ["cofactors / metals (+CCD)",
                  "stereochem &#183; covalent intermediate",
                  "substrate (+reactive atom, SMILES) &#183; Michaelis PDB",
                  "SN specialization: Nu / electrophile / LG &#183; clean_inline?"]),
        _lines(["status:  ok  |  low_confidence (+warning)  &#183;  papers_tried"],
               size=10, color=FAINT))
    g.node("out", _card("Output: mechanism facts  (schema-validated)", out_body, A))

    g.edge("in", "src")
    g.edge("in", "loop", style="dashed", color=FAINT, label=" paper supplied ",
           fontsize="9", fontcolor=FAINT, constraint="false")
    g.edge("merge", "loop")
    g.edge("fallback", "loop", style="dashed", color=FAINT)
    g.edge("loop", "gate")
    g.edge("gate", "loop", style="dashed", color=FAINT, label=" weak &#8594; next paper ",
           fontsize="9", fontcolor=FAINT, constraint="false")
    g.edge("gate", "out", label=" grounded / exhausted ", fontsize="9", fontcolor=MUTED)
    return g


# --------------------------------------------------------------------------- #
#  extract_experimental_measurements                                          #
# --------------------------------------------------------------------------- #
def build_measurement() -> Digraph:
    A = "#0d9488"   # teal — feeds structural-modeling / Report
    g = _base("measurement_extraction_flow", "extract_experimental_measurements",
              "structured kinetics / activity rows from tables &amp; figures, with provenance")

    g.node("in", _card("Input",
                       _lines(["identifier:  DOI / PMID / PMCID / title",
                               "context_hint &#183; try_images (default on)"]), A))
    g.node("fetch", _card("resolve_paper &#8594; fetch open-access full text",
                         _lines(["fetch_jats_or_text"]), A))

    with g.subgraph(name="cluster_kind") as c:
        _cluster(c, "route by content type  (structure-preserving)", A)
        c.node("xml", _card("JATS XML",
                           _lines(["parse_jats_tables &#8594; header + rows kept",
                                   "parse_jats_figures &#183; flag image-only"]), A))
        c.node("pdf", _card("PDF",
                           _lines(["render cue-bearing pages (&#8804;6)", "&#8594; vision"]), A))
        c.node("flat", _card("no JATS",
                            _lines(["large flat-text pass", "(fallback)"], color=MUTED), FAINT))

    g.node("jobs", _card("Parallel extraction &#183; &#8804;6 workers",
                        _lines(["text tables &#8594; LLM",
                                "image tables / figures &#8594; vision LLM"]), A))

    out_body = _stack(
        _two_col(["metric:  kcat &#183; Km &#183; kcat/Km",
                  "specific_activity &#183; conversion%",
                  "yield% &#183; titer &#183; Tm &#183; pH/T optimum"],
                 ["value &#183; units &#183; enzyme",
                  "substrate &#183; conditions",
                  "source table / figure"]),
        _lines(["one row per measurement  &#183;  count &#183; status"],
               size=10, color=FAINT))
    g.node("out", _card("Output: structured measurements  (+ provenance)", out_body, A))

    g.edge("in", "fetch")
    g.edge("fetch", "xml")
    g.edge("fetch", "pdf")
    g.edge("fetch", "flat", style="dashed", color=FAINT)
    g.edge("xml", "jobs")
    g.edge("pdf", "jobs")
    g.edge("flat", "jobs", style="dashed", color=FAINT)
    g.edge("jobs", "out")
    return g


def main():
    for builder, stem in ((build_mechanism, "mechanism_extraction_flow"),
                          (build_measurement, "measurement_extraction_flow")):
        g = builder()
        out = HERE / stem
        g.render(str(out), format="png", cleanup=True)
        print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
