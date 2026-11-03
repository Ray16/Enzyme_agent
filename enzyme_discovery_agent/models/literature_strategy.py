"""Complementary literature search routes derived from the target and evidence.

No answer enzymes are supplied. A negative keyword search closes one route,
not its chemical hypothesis: other sources, broader terms and citation graphs
remain eligible. All progress is durable across process restarts.
"""

import re
from pathlib import Path

from enzyme_discovery_agent.models.target_reasoning import target_claims
from enzyme_discovery_agent.run_layout import run_file

SOURCES = ("search_pubmed", "search_openalex", "search_crossref", "search_europepmc", "search_semantic_scholar")


def target_routes(run_dir):
    from enzyme_discovery_agent.models.family_learning import _read
    root = Path(run_dir)
    context = _read(run_file(root, "target_context.json"), {})
    direction, bonds, _ = target_claims(root)
    if not bonds and not direction:
        return []
    bond = " ".join(sorted(bonds)) or "molecular transformation"
    query = context.get("query") or ""
    application = re.search(r"\bfor\s+(.+?)(?:\s+(?:production|synthesis|manufactur\w*)|$)", query, re.I)
    application = re.sub(r"[\d,.-]+", " ", application.group(1)).strip() if application else ""
    direction_terms = {"formation": ["formation", "synthesis"],
                       "cleavage": ["cleavage", "degradation"]}.get(direction, [direction or "catalysis"])
    routes = []
    if application:
        routes.append((f"{application} enzymatic {bond} {direction_terms[0]}", "application", True))
    routes.append((f"enzymatic {bond} {direction_terms[0]}", "mechanism", True))
    chemistry = _read(run_file(root, "target_reasoning.json"), {})
    groups = [str(value).replace("_", " ") for value in chemistry.get("reactive_groups", [])]
    if groups:
        routes.append((f"{' '.join(groups)} enzymatic {bond} {direction_terms[0]}", "substrate_class", False))
    routes.extend((f"enzyme {bond} {term} substrate scope", "mechanism", False) for term in direction_terms)
    for category in chemistry.get("activation_categories", []) + chemistry.get("cofactor_requirements", []):
        if isinstance(category, str) and category.strip():
            routes.append((f"{category.strip()} enzymatic {bond} {direction_terms[0]}", "derived_mechanism", False))
    if query:
        cleaned = re.sub(r"\b(find|identify|that|which|for|production|candidates?)\b", " ", query, flags=re.I)
        routes.append((" ".join(cleaned.split()), "target", False))
    return routes


def query_variants(query):
    """Relax exact combinations and normalize common chemistry synonyms."""
    words = query.split()
    variants = [query]
    if len(words) >= 5:
        variants.append(" ".join(words[:3]))
        variants.append(" ".join(words[-3:]))
        variants.append(" ".join(words[::2]))
    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))


def complementary_actions(run_dir, tool_names, hypotheses):
    """Independent baseline first, then complementary routes for every hypothesis."""
    sources = [name for name in SOURCES if name in tool_names]
    if not sources:
        return []
    tasks = []
    for query, lane, baseline in target_routes(run_dir):
        for source in sources:
            tasks.append({"tool": source, "args": {"query": query, "max_results": 15},
                          "reason": f"Independent {lane} search across literature sources",
                          "selection": "baseline" if baseline and source in sources[:2] else "exploration",
                          "lane": lane})
            if source == "search_openalex":
                tasks.append({"tool": source, "args": {"query": query, "max_results": 15, "sort_by_citations": False,"sort_order":"recent"},
                              "reason": f"Recent literature coverage for {lane}", "selection": "exploration", "lane": lane})
    # Low model promise is a hypothesis, not evidence that a branch is closed.
    for family in hypotheses:
        for term in family.get("literature_terms") or []:
            for query in query_variants(term):
                for source in sources:
                    tasks.append({"tool": source, "args": {"query": query, "max_results": 15},
                                  "reason": f"Complementary search for {family.get('name', 'mechanism')}",
                                  "selection": "exploration", "lane": "hypothesis"})
    return sorted(tasks, key=lambda t: t["selection"] != "baseline")


def citation_actions(run_dir, tool_names):
    """Follow saved papers explicitly, preserving full records of discovered refs."""
    if not {"follow_paper_citations", "fetch_paper_metadata"} & tool_names:
        return []
    from enzyme_discovery_agent.tools.literature import _recover_literature_papers
    from enzyme_discovery_agent.tools.paper_relevance import rank_papers_by_enzymology
    from enzyme_discovery_agent.models.family_learning import _read
    root = Path(run_dir)
    from enzyme_discovery_agent.tools.literature import dedupe_and_merge
    papers, _ = dedupe_and_merge(_recover_literature_papers(str(root)))
    try:
        papers = rank_papers_by_enzymology(root, papers, fallback=lambda: papers)
    except (ValueError, TypeError):
        pass
    context = _read(run_file(root, "target_context.json"), {})
    actions = []
    for paper in papers:
        if "fetch_paper_metadata" in tool_names:
            for relation in (paper.get("related_identifiers") or {}).values():
                for related in relation:
                    if related.get("id-type") == "doi" and related.get("id"):
                        actions.append({"tool": "fetch_paper_metadata", "args": {"doi": related["id"]},
                                        "reason": "Resolve the article explicitly linked by a related publication record",
                                        "selection": "citation", "lane": "parent_article"})
            if not paper.get('abstract') and (paper.get('pmid') or paper.get('doi')):
                identifier={'pmid':str(paper['pmid'])} if paper.get('pmid') else {'doi':paper['doi']}
                actions.append({'tool':'fetch_paper_metadata','args':identifier,
                    'reason':'Resolve the missing abstract of a retrieved citation before judging its relevance.',
                    'selection':'citation','lane':'citation_metadata'})
        if "follow_paper_citations" not in tool_names:
            continue
        if not (paper.get("doi") or paper.get("pmid")):
            continue
        actions.append({"tool": "follow_paper_citations", "args": {
            "doi": paper.get("doi") or "", "pmid": str(paper.get("pmid") or ""),
            "title": paper.get("title") or "", "query": context.get("query") or "enzyme catalysis",
        }, "reason": f"Follow primary references and newer citations of {paper.get('title', '')}",
            "selection": "citation", "lane": "citation"})
    return actions


def coverage_summary(run_dir, ledger):
    from enzyme_discovery_agent.tools.literature import _recover_literature_papers
    from enzyme_discovery_agent.models.family_learning import _read
    root = Path(run_dir)
    papers = _recover_literature_papers(str(root))
    reads = _read(root / "artifacts" / "fulltext" / "manifest.json", [])
    return {"paper_records": len(papers), "full_texts_acquired": len(reads),
            "sources_tried": sorted({r.get("tool") for r in ledger.values() if r.get("tool") in SOURCES}),
            "citation_actions": sum(r.get("selection") == "citation" for r in ledger.values()),
            "failed_routes": sum(r.get("status") == "error" for r in ledger.values()),
            "assessment": "Search frontier status is not proof that all relevant families or candidates were found"}
