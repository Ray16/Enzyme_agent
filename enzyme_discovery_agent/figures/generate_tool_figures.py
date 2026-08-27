#!/usr/bin/env python3
"""Render internal-flow figures for the two literature evidence-extractor tools:

  * extract_reaction_mechanism      -> mechanism_extraction_flow.{png,svg}
  * extract_experimental_measurements -> measurement_extraction_flow.{png,svg}

Each figure shows how the tool works internally (discovery -> read -> extract ->
score/validate -> output), faithfully to the code in
``enzyme_discovery_agent/tools/{mechanism,measurement}_extraction.py``.

Generic and answer-agnostic (firewall-safe): the diagrams describe tool
mechanics only, never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_tool_figures.py
Outputs (next to this script): the two *_flow.png and *_flow.svg files.
"""
from pathlib import Path

from graphviz import Digraph

HERE = Path(__file__).resolve().parent


def _tint(hex_color: str, amount: float = 0.90) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _card(title: str, lines, accent: str, title_size: int = 15) -> str:
    """HTML-like label: bold title over muted descriptor lines."""
    body = "".join(
        f'<tr><td><font point-size="10.5" color="#475569">{ln}</font></td></tr>'
        for ln in lines
    )
    return (
        f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="1">'
        f'<tr><td><font point-size="{title_size}" color="#0f172a"><b>{title}</b></font></td></tr>'
        f'{body}</table>>'
    )


def _base(name: str) -> Digraph:
    g = Digraph(name)
    g.attr(rankdir="TB", bgcolor="white", dpi="220", ranksep="0.42",
           nodesep="0.34", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", shape="box", style="filled,rounded",
           margin="0.16,0.10", penwidth="2.0")
    g.attr("edge", color="#94a3b8", penwidth="2.2", arrowsize="0.9")
    return g


# --------------------------------------------------------------------------- #
#  extract_reaction_mechanism                                                  #
# --------------------------------------------------------------------------- #
def build_mechanism() -> Digraph:
    A = "#7c3aed"  # purple — feeds Evaluate/Rank
    g = _base("mechanism_extraction_flow")
    g.attr(label=(
        '<<font point-size="19" color="#0f172a"><b>extract_reaction_mechanism</b></font>'
        '<br/><font point-size="11" color="#64748b">'
        'grounded, reaction-class-agnostic mechanism facts from primary literature</font>>'),
        labelloc="t", labeljust="c")

    g.node("in", _card("Input", ["enzyme name / EC / accession",
                                 "optional: specific paper (DOI/PMID/PMCID)"], A, 14),
           fillcolor=_tint(A, 0.94), color=A)

    # Discovery: parallel federated search
    with g.subgraph(name="cluster_disc") as c:
        c.attr(label="  candidate discovery (skipped if a paper is supplied)",
               fontname="Helvetica", fontsize="11", color=_tint(A, 0.55),
               style="rounded", penwidth="1.6")
        c.node("src", _card("Federated search  &#183;  4 sources in parallel",
                            ["PubMed &#183; OpenAlex &#183; Europe PMC &#183; Semantic Scholar",
                             "class-agnostic query (no &#8216;nucleophile&#8217; bias)"], A),
               fillcolor="white", color=A)
        c.node("merge", _card("consolidate_pool",
                              ["cross-source dedup / merge (back-fills PMCID)",
                               "semantic RRF rerank &#8594; ranked candidates"], A),
               fillcolor="white", color=A)
        c.node("fallback", _card("fallback",
                                 ["Europe PMC keyword search",
                                  "+ mechanism/structure keyword ranking"], "#94a3b8"),
               fillcolor="#f8fafc", color="#cbd5e1")
        c.edge("src", "merge")
        c.edge("src", "fallback", style="dashed", color="#cbd5e1",
               label="  stack down", fontsize="9", fontcolor="#94a3b8")

    # Extraction loop
    g.node("loop", _card("For each candidate, best-first  (early-stop when confident)",
                         ["fetch full text: JATS&#8594;text  or  PDF&#8594;text",
                          "LLM extract mechanism JSON (14k-char excerpt)",
                          "score groundedness &#8594; keep best"], A),
           fillcolor=_tint(A, 0.90), color=A, penwidth="2.6")

    g.node("gate", _card("Groundedness gate  (_score_facts)",
                         ["named participants + bonds + residues/cofactors?",
                          "penalise &#8216;no detail&#8217; disclaimers &amp; bare-water Nu"], A),
           fillcolor=_tint(A, 0.94), color=A)

    g.node("out", _card("Output: mechanism facts  (schema-validated)",
                        ["mechanism_class &#183; chemical_step &#183; bonds broken/formed",
                         "participants &#183; reactive_pair &#183; catalytic_residues",
                         "cofactors/metals (+CCD) &#183; stereochem &#183; covalent intermed.",
                         "substrate (+reactive atom, SMILES) &#183; Michaelis PDB",
                         "SN specialization: Nu / electrophile / LG, clean_inline?",
                         "status: ok  |  low_confidence (+warning) &#183; papers_tried"], A, 14),
           fillcolor=_tint(A, 0.90), color=A, penwidth="2.6")

    g.edge("in", "src")
    g.edge("in", "loop", style="dashed", label="  paper supplied",
           fontsize="9", fontcolor="#94a3b8", constraint="false")
    g.edge("merge", "loop")
    g.edge("fallback", "loop", style="dashed", color="#cbd5e1")
    g.edge("loop", "gate")
    g.edge("gate", "loop", style="dashed", label="  weak &#8594; next paper",
           fontsize="9", fontcolor="#94a3b8", constraint="false")
    g.edge("gate", "out", label="  grounded / exhausted", fontsize="9", fontcolor="#64748b")
    return g


# --------------------------------------------------------------------------- #
#  extract_experimental_measurements                                          #
# --------------------------------------------------------------------------- #
def build_measurement() -> Digraph:
    A = "#0d9488"  # teal — feeds structural-modeling / Report
    g = _base("measurement_extraction_flow")
    g.attr(label=(
        '<<font point-size="19" color="#0f172a"><b>extract_experimental_measurements</b></font>'
        '<br/><font point-size="11" color="#64748b">'
        'structured kinetics / activity rows from tables &amp; figures, with provenance</font>>'),
        labelloc="t", labeljust="c")

    g.node("in", _card("Input", ["identifier: DOI / PMID / PMCID / title",
                                 "context_hint &#183; try_images (default on)"], A, 14),
           fillcolor=_tint(A, 0.94), color=A)
    g.node("fetch", _card("resolve_paper &#8594; fetch open-access full text",
                          ["fetch_jats_or_text"], A),
           fillcolor=_tint(A, 0.94), color=A)

    # Branch on content kind
    with g.subgraph(name="cluster_kind") as c:
        c.attr(label="  route by content type (structure-preserving)",
               fontname="Helvetica", fontsize="11", color=_tint(A, 0.55),
               style="rounded", penwidth="1.6")
        c.node("xml", _card("JATS XML",
                            ["parse_jats_tables &#8594; header+rows kept",
                             "parse_jats_figures &#183; flag image-only"], A),
               fillcolor="white", color=A)
        c.node("pdf", _card("PDF",
                            ["render cue-bearing pages (&#8804;6)",
                             "&#8594; vision"], A),
               fillcolor="white", color=A)
        c.node("flat", _card("no JATS",
                             ["large flat-text pass", "(fallback)"], "#94a3b8"),
               fillcolor="#f8fafc", color="#cbd5e1")

    g.node("jobs", _card("Parallel extraction  (&#8804;6 workers)",
                         ["text tables &#8594; LLM",
                          "image tables / figures &#8594; vision LLM"], A),
           fillcolor=_tint(A, 0.90), color=A, penwidth="2.6")

    g.node("out", _card("Output: one row per measurement  (+ provenance)",
                        ["metric: kcat &#183; Km &#183; kcat/Km &#183; specific_activity",
                         "conversion% &#183; yield% &#183; titer &#183; Tm &#183; pH/T optimum",
                         "value &#183; units &#183; enzyme &#183; substrate &#183; conditions",
                         "source table / figure  &#183;  count &#183; status"], A, 14),
           fillcolor=_tint(A, 0.90), color=A, penwidth="2.6")

    g.edge("in", "fetch")
    g.edge("fetch", "xml")
    g.edge("fetch", "pdf")
    g.edge("fetch", "flat", style="dashed", color="#cbd5e1")
    g.edge("xml", "jobs")
    g.edge("pdf", "jobs")
    g.edge("flat", "jobs", style="dashed", color="#cbd5e1")
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
