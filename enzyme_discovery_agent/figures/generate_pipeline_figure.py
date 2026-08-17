#!/usr/bin/env python3
"""Render the enzyme_discovery_agent workflow pipeline as a diagram.

Generic and answer-agnostic (firewall-safe): it describes the MECHANISM of the
pipeline — LLM proposes strategies, deterministic gates + tools find entities —
never a specific target or answer enzyme.

Usage:  python enzyme_discovery_agent/figures/generate_pipeline_figure.py
Outputs (next to this script): pipeline_workflow.png and pipeline_workflow.svg
"""
from pathlib import Path

from graphviz import Digraph

OUT = Path(__file__).resolve().parent / "pipeline_workflow"

# Node role palette
C = {
    "input":   ("#1f4e79", "#dbe9f6"),   # target setup
    "llm":     ("#7a3b00", "#ffe6cc"),   # LLM reasoning (proposes strategy)
    "gate":    ("#4b0082", "#e9d5ff"),   # deterministic pipeline gate
    "tool":    ("#0b6b3a", "#d6f0e0"),   # tool / external database
    "score":   ("#8a1f1f", "#ffd9d9"),   # deterministic scoring / selection
    "artifact":("#333333", "#eeeeee"),   # persisted artifact / output
}


def node(g, name, label, role, shape="box"):
    fg, bg = C[role]
    g.node(name, label, shape=shape, style="filled,rounded" if shape == "box" else "filled",
           fillcolor=bg, color=fg, fontcolor=fg, penwidth="1.6",
           fontname="Helvetica", fontsize="11")


def build() -> Digraph:
    g = Digraph("enzyme_discovery_pipeline", format="png")
    g.attr(rankdir="TB", bgcolor="white", splines="spline",
           nodesep="0.35", ranksep="0.55", fontname="Helvetica")
    g.attr("edge", color="#666666", penwidth="1.3", arrowsize="0.8")

    # ---- 1. Target setup ----
    with g.subgraph(name="cluster_setup") as c:
        c.attr(label="1  TARGET SETUP", style="rounded", color="#1f4e79",
               fontcolor="#1f4e79", fontname="Helvetica-Bold", fontsize="13")
        node(c, "query", "User query + substrate(s)\n(natural language + SMILES)", "input", "note")
        node(c, "profile", "Resolve substrates → SMILES\nbuild target profile\n(bond type + direction)", "llm")
        c.edge("query", "profile")

    # ---- 2. Discovery loop ----
    with g.subgraph(name="cluster_discovery") as c:
        c.attr(label="2  DISCOVERY  (agentic loop: LLM proposes strategy, gates + tools find entities)",
               style="rounded", color="#4b0082", fontcolor="#4b0082",
               fontname="Helvetica-Bold", fontsize="13")
        node(c, "loop", "LLM agent loop\n(proposes tool calls each iteration)", "llm")
        node(c, "litsearch", "Literature search\nPubMed · OpenAlex · EuropePMC · S2\n+ semantic re-rank → one pool", "tool")
        node(c, "harvest", "Gate: literature harvest (lean)\nread top-N open-access papers →\nLLM-extract NAMED / CITED enzymes", "gate")
        node(c, "families", "Gate: mechanism families\nLLM derives distinct folds\n(direction + free-substrate filtered)", "gate")
        node(c, "survey", "Gate: InterPro family survey\nsurvey each fold's members", "gate")
        node(c, "fit", "Substrate-fit scoring\none-to-one assignment,\npromote EVERY relevant member", "score")
        node(c, "dbsearch", "Database search\nUniProt / InterPro\n(homolog + family expansion)", "tool")
        node(c, "registry", "Candidate registry\n(discovered enzymes)", "artifact", "cylinder")

        c.edge("loop", "litsearch")
        c.edge("litsearch", "harvest")
        c.edge("harvest", "families")
        c.edge("families", "survey")
        c.edge("survey", "fit")
        c.edge("fit", "registry", label="promote")
        c.edge("loop", "dbsearch", style="dashed")
        c.edge("dbsearch", "registry")
        c.edge("harvest", "registry", label="seeds", style="dashed")
        c.edge("registry", "loop", label="expand around seeds",
               style="dashed", constraint="false")

    # ---- 3. Screen + evaluate ----
    with g.subgraph(name="cluster_eval") as c:
        c.attr(label="3  SCREEN + EVALUATE", style="rounded", color="#8a1f1f",
               fontcolor="#8a1f1f", fontname="Helvetica-Bold", fontsize="13")
        node(c, "screen", "Ortholog collapse →\nscreen EVERY gene representative\n(direction gate)", "score")
        node(c, "evaluate", "Evaluate all reps (parallel, no LLM)\nUniProt · InterPro · PDB · ChEBI\nsubstrate similarity, warnings", "tool")
        c.edge("screen", "evaluate")

    # ---- 4. Rank ----
    with g.subgraph(name="cluster_rank") as c:
        c.attr(label="4  RANK", style="rounded", color="#8a1f1f",
               fontcolor="#8a1f1f", fontname="Helvetica-Bold", fontsize="13")
        node(c, "detrank", "Deterministic mechanistic spine\n+ family-relevance binding\n(off-target folds hard-gated down)", "score")
        node(c, "llmrank", "LLM focus-rank\n(reasons over top shortlist)", "llm")
        node(c, "pass1", "pass1_ranking.json", "artifact", "cylinder")
        c.edge("detrank", "llmrank")
        c.edge("llmrank", "pass1")

    # ---- 5. Report ----
    with g.subgraph(name="cluster_report") as c:
        c.attr(label="5  REPORT", style="rounded", color="#333333",
               fontcolor="#333333", fontname="Helvetica-Bold", fontsize="13")
        node(c, "reports", "Two-pass reports\nconfidence + explorative\n+ citation verification", "llm")
        node(c, "outputs", "report_*.md · ranked_candidates.csv\nfamily-fit plot · structures", "artifact", "folder")
        c.edge("reports", "outputs")

    # ---- cross-stage flow ----
    g.edge("profile", "loop", label="target profile")
    g.edge("registry", "screen", label="candidates")
    g.edge("evaluate", "detrank", label="evaluations")
    g.edge("pass1", "reports")

    # ---- legend ----
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", style="rounded", color="#999999",
               fontcolor="#555555", fontname="Helvetica-Bold", fontsize="11")
        for key, txt in [("llm", "LLM reasoning\n(proposes strategy)"),
                         ("gate", "deterministic gate"),
                         ("tool", "tool / external DB"),
                         ("score", "deterministic scoring"),
                         ("artifact", "artifact / output")]:
            node(c, f"lg_{key}", txt, key)
        # keep legend nodes in a row, invisible ordering edges
        c.attr(rank="same")
        c.edge("lg_llm", "lg_gate", style="invis")
        c.edge("lg_gate", "lg_tool", style="invis")
        c.edge("lg_tool", "lg_score", style="invis")
        c.edge("lg_score", "lg_artifact", style="invis")

    return g


def main():
    g = build()
    g.render(str(OUT), format="png", cleanup=True)
    g.render(str(OUT), format="svg", cleanup=True)
    print(f"wrote {OUT}.png and {OUT}.svg")


if __name__ == "__main__":
    main()
