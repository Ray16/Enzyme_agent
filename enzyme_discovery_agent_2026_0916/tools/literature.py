"""Literature search tools: PubMed and OpenAlex with semantic re-ranking.

All search functions support semantic re-ranking using a sentence-transformer
model (nomic-embed-text-v1.5) to score title+abstract similarity against the
query intent. This surfaces relevant papers that keyword matching would miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")


# ---------------------------------------------------------------------------
# Request-identity helpers (API keys + polite-pool identifiers)
# ---------------------------------------------------------------------------

def _ncbi_params(extra: dict) -> dict:
    """Merge NCBI etiquette (tool/email) and an api_key when configured.

    A free NCBI api_key raises the E-utilities limit from 3 to 10 req/s; NCBI
    asks every request to carry ``tool`` and ``email``.  Caller-supplied keys
    in ``extra`` win over the defaults.
    """
    params = {"tool": config.NCBI_TOOL, "email": config.CONTACT_EMAIL}
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    params.update(extra)
    return params


def _openalex_auth_qs() -> str:
    """Return an ``&api_key=...`` URL fragment for OpenAlex, or "" when unset.

    OpenAlex retired the mailto polite pool (2026-02-13); without a key the
    daily budget is ~100 list requests before HTTP 409.
    """
    return f"&api_key={config.OPENALEX_API_KEY}" if config.OPENALEX_API_KEY else ""


# ---------------------------------------------------------------------------
# Semantic re-ranking
# ---------------------------------------------------------------------------
_RERANK_MODEL = None
_RERANK_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
_RERANK_MODEL_LOCK = threading.Lock()


def _get_rerank_model():
    """Lazy-load the sentence-transformer model (thread-safe singleton)."""
    global _RERANK_MODEL
    if not config.ENABLE_SEMANTIC_RERANK:
        # Do not even import sentence_transformers in the default workflow.
        # The caller preserves PubMed/OpenAlex's normal relevance order.
        return None
    if _RERANK_MODEL is not None:
        return _RERANK_MODEL
    with _RERANK_MODEL_LOCK:
        if _RERANK_MODEL is not None:
            return _RERANK_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading semantic re-ranking model on CPU: %s",
                _RERANK_MODEL_NAME,
            )
            _RERANK_MODEL = SentenceTransformer(
                _RERANK_MODEL_NAME, trust_remote_code=True,
                device="cpu",
                config_kwargs={"use_flash_attn": False},
            )
            return _RERANK_MODEL
        except Exception as e:
            # Reaching here means the feature is ENABLED but the model could not
            # load -- a real misconfiguration, not a routine skip.  Be loud and
            # actionable rather than silently degrading to native order.
            logger.error(
                "Semantic rerank is ENABLED but the model failed to load; "
                "falling back to native retrieval order. Install the optional "
                "dependency (pip install sentence-transformers). Cause: %s", e,
            )
            return None


def cleanup_rerank_model() -> None:
    """Release the semantic re-ranking model and free GPU memory."""
    global _RERANK_MODEL
    with _RERANK_MODEL_LOCK:
        if _RERANK_MODEL is not None:
            _RERANK_MODEL = None
            logger.info("Released semantic re-ranking model")


def _semantic_rerank(
    query: str,
    papers: list[dict],
    max_results: int,
    min_score: float = config.RERANK_MIN_SCORE,
    title_key: str = "title",
    abstract_key: str = "abstract",
) -> list[dict]:
    """Re-rank papers by fusing native retrieval order with semantic similarity.

    The input ``papers`` arrive in the source's own relevance order (PubMed,
    OpenAlex, Europe PMC, Semantic Scholar all pre-rank).  Rather than discard
    that signal, we compute a semantic (cosine) rank from embeddings and fuse
    the two with Reciprocal Rank Fusion, so a paper strong on *either* axis
    surfaces.  RRF governs ORDER only; inclusion is gated by ``min_score`` on
    the cosine score -- a stable [0,1] scale -- so the threshold's meaning is
    unchanged by fusion (the two can be recalibrated together).

    Degrades gracefully to native order when the embedder is unavailable
    (feature disabled or the optional dependency is missing).
    """
    if not papers:
        return []
    model = _get_rerank_model()
    if model is None:
        # Expected no-op: rerank disabled or sentence-transformers absent.
        return papers[:max_results]

    # A failure computing embeddings/cosine is a genuine bug, NOT the expected
    # "embedder unavailable" path -- surface it loudly (with traceback) instead
    # of letting it masquerade as a graceful fallback.
    try:
        sims = _semantic_scores(model, query, papers, title_key, abstract_key)
    except Exception as e:
        logger.error("Semantic rerank computation failed; returning native "
                     "order: %s", e, exc_info=True)
        return papers[:max_results]

    return _rrf_fuse(papers, sims, max_results, min_score)


# Embedding cache: within (and across) searches in a campaign the same abstract
# recurs constantly.  Cache normalized embeddings by (model, text-hash) so each
# unique document is encoded once, not once per search that surfaces it.
_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}
_EMBED_CACHE_LOCK = threading.Lock()


def _embed_texts(model, texts: list[str]):
    """Return normalized embeddings for ``texts``, encoding only cache misses."""
    import numpy as np
    out: list[list[float] | None] = [None] * len(texts)
    misses, miss_slots = [], []
    for i, t in enumerate(texts):
        h = (_RERANK_MODEL_NAME, hashlib.sha1(t.encode("utf-8")).hexdigest())
        cached = _EMBED_CACHE.get(h)
        if cached is not None:
            out[i] = cached
        else:
            misses.append(t)
            miss_slots.append((i, h))
    if misses:
        emb = np.asarray(model.encode(misses, normalize_embeddings=True,
                                      batch_size=32, show_progress_bar=False))
        with _EMBED_CACHE_LOCK:
            for j, (i, h) in enumerate(miss_slots):
                vec = emb[j].tolist()
                _EMBED_CACHE[h] = vec
                out[i] = vec
    return np.asarray(out)


def _semantic_scores(model, query, papers, title_key, abstract_key):
    """Per-paper cosine similarity to the query.

    Embeddings are normalized, so cosine is a plain dot product -- computed with
    numpy (already present via pandas), which removes the previous undeclared
    scikit-learn dependency entirely.  Encoding is cached per unique text.
    """
    query_text = f"search_query: {query}"
    doc_texts = [
        f"search_document: {p.get(title_key, '')}. {p.get(abstract_key, '')[:500]}"
        for p in papers
    ]
    q_emb = _embed_texts(model, [query_text])[0]
    p_emb = _embed_texts(model, doc_texts)
    return p_emb @ q_emb


def _rrf_fuse(papers, sims, max_results, min_score, k: int = 60):
    """Fuse native retrieval order with semantic (cosine) order via Reciprocal
    Rank Fusion.

    RRF sets ORDER (a paper high on *either* the source's native rank or the
    cosine rank floats up); ``min_score`` gates INCLUSION on the cosine scale.
    ``k`` is the standard RRF damping constant.  Attaches ``semantic_score`` and
    ``rrf_score`` to each returned paper for auditability.
    """
    n = len(papers)
    # native rank = input position (0 = most relevant per the source);
    # cosine rank = position when sorted by similarity, descending.
    cosine_order = sorted(range(n), key=lambda i: sims[i], reverse=True)
    cosine_rank = {i: r for r, i in enumerate(cosine_order)}

    fused = []
    for i, paper in enumerate(papers):
        rrf = 1.0 / (k + i) + 1.0 / (k + cosine_rank[i])
        pc = dict(paper)
        pc["semantic_score"] = round(float(sims[i]), 4)
        pc["rrf_score"] = round(float(rrf), 6)
        fused.append(pc)

    fused.sort(key=lambda p: p["rrf_score"], reverse=True)
    kept = [p for p in fused if p["semantic_score"] >= min_score]
    # Never starve to empty purely on the floor: an over-tight threshold should
    # degrade to the fused ranking, not return nothing.
    if not kept:
        kept = fused
    return kept[:max_results]


# ---------------------------------------------------------------------------
# Pool consolidation: cross-source dedup + record merge + single rerank
# ---------------------------------------------------------------------------
# Each search tool reranks only its own results, and the same paper is often
# returned by several sources -- so it gets embedded repeatedly and the LLM sees
# near-duplicates.  Consolidating the accumulated pool ONCE (dedup on a shared
# key, merge fields across sources, then a single global RRF rerank) is cheaper
# (one embedding pass, cache-backed) and ranks better (global order).

def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _paper_key(paper: dict):
    """Stable identity for cross-source dedup: DOI, else PMID, else norm title."""
    doi = (paper.get("doi") or "").lower().strip()
    if doi:
        return ("doi", doi)
    pmid = str(paper.get("pmid") or "").strip()
    if pmid:
        return ("pmid", pmid)
    nt = _normalize_title(paper.get("title", ""))
    if nt:
        return ("title", nt)
    return None


def _merge_records(a: dict, b: dict) -> dict:
    """Merge duplicate ``b`` into ``a``: fill empty fields, keep longer abstract."""
    merged = dict(a)
    for k, v in b.items():
        if k in ("source", "sources"):
            continue  # provenance handled separately in dedupe_and_merge
        av = merged.get(k)
        if k == "abstract":
            if len(v or "") > len(av or ""):
                merged[k] = v
        elif (av in (None, "", 0, [])) and v not in (None, "", 0, []):
            merged[k] = v
    return merged


def dedupe_and_merge(papers: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicate across sources and merge records, preserving first-seen order.

    Returns (unique_papers, stats).  Each unique paper gains a ``sources`` list
    (union of the sources that surfaced it), so provenance survives the merge --
    e.g. a paper from PubMed (no PMCID) merged with its Europe PMC record (has
    PMCID) ends up routable to full text.
    """
    merged: dict = {}
    order: list = []
    src: dict = {}
    unkeyable: list[dict] = []

    def _add_src(key, paper):
        s = paper.get("sources") or paper.get("source")
        if isinstance(s, list):
            src[key].update(x for x in s if x)
        elif s:
            src[key].add(s)

    for p in papers:
        key = _paper_key(p)
        if key is None:
            unkeyable.append(dict(p))
            continue
        if key in merged:
            merged[key] = _merge_records(merged[key], p)
        else:
            merged[key] = dict(p)
            order.append(key)
            src[key] = set()
        _add_src(key, p)

    unique = []
    for key in order:
        rec = merged[key]
        rec["sources"] = sorted(src[key])
        unique.append(rec)
    unique.extend(unkeyable)

    stats = {
        "raw": len(papers),
        "unique": len(unique),
        "duplicates_merged": len(papers) - len(unique),
    }
    return unique, stats


def consolidate_pool(query: str, papers: list[dict], max_results: int = 25,
                     min_score: float = config.RERANK_MIN_SCORE) -> dict:
    """Dedup+merge a pooled paper set, then a single global RRF rerank.

    Returns ``{"ranked": [...], "funnel": {...}}`` where the funnel carries
    PRISMA-style counts for auditability.
    """
    unique, stats = dedupe_and_merge(papers)
    ranked = _semantic_rerank(query, unique, max_results=max_results,
                              min_score=min_score)
    funnel = {
        "raw": stats["raw"],
        "unique": stats["unique"],
        "duplicates_merged": stats["duplicates_merged"],
        "kept": len(ranked),
    }
    return {"ranked": ranked, "funnel": funnel}


def _recover_literature_papers(run_dir: str | None) -> list[dict]:
    """Recover papers already collected this run from literature artifacts."""
    if not run_dir:
        return []
    from pathlib import Path

    recovered: list[dict] = []
    for path in sorted((Path(run_dir) / "artifacts" / "literature").glob("*.json")):
        try:
            artifact = json.loads(path.read_text())
            data = artifact.get("result", {}).get("data", [])
            if isinstance(data, list):
                recovered.extend(item for item in data if isinstance(item, dict))
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    seen = set()
    papers: list[dict] = []
    for paper in recovered:
        key = (paper.get("pmid"), paper.get("doi"), paper.get("title"))
        if key in seen:
            continue
        seen.add(key)
        papers.append(paper)
    return papers


@tool(
    description=(
        "Consolidate all literature gathered so far this run: deduplicate the "
        "same paper across sources (merging fields, e.g. filling a PMCID from "
        "Europe PMC onto a PubMed record), then produce a single globally "
        "re-ranked shortlist. Call this after several searches, before deep "
        "reading, to remove near-duplicates and get one clean ranked pool. "
        "Safe to call repeatedly as the pool grows."
    ),
    # 'reasoning' (not 'literature') so repeated consolidation over a growing
    # pool is never blocked by the literature-search query-dedup, and so it is
    # not miscounted as a fresh literature-search round.
    category="reasoning",
    parallel_safe=True,
    injected_params=["run_dir"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "query": {
            "type": "string",
            "description": "Campaign intent to rank the consolidated pool against.",
        },
        "max_results": {
            "type": "integer",
            "description": "Size of the ranked shortlist to return (default 25).",
            "default": 25,
        },
    },
)
def consolidate_literature(query: str, run_dir: str | None = None,
                           max_results: int = 25) -> dict:
    """Dedup+merge accumulated literature and return one globally-ranked pool."""
    papers = _recover_literature_papers(run_dir)
    if not papers:
        return {
            "status": "partial",
            "summary": "No accumulated literature to consolidate yet.",
            "data": [],
            "count": 0,
            "suggestions": ["Run literature searches first, then consolidate."],
        }
    result = consolidate_pool(query, papers, max_results=max_results)
    f = result["funnel"]
    logger.info("PRISMA funnel: raw=%d -> unique=%d (merged %d dups) -> kept=%d",
                f["raw"], f["unique"], f["duplicates_merged"], f["kept"])
    return {
        "status": "success",
        "summary": (f"Consolidated {f['raw']} hits -> {f['unique']} unique "
                    f"(-{f['duplicates_merged']} dup) -> {f['kept']} ranked"),
        "data": result["ranked"],
        "funnel": f,
        "count": f["kept"],
    }


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def _parse_pubmed_article(article_elem) -> dict | None:
    """Parse a single PubmedArticle XML element."""
    medline = article_elem.find(".//MedlineCitation")
    if medline is None:
        return None

    pmid_elem = medline.find("PMID")
    pmid = pmid_elem.text if pmid_elem is not None else ""

    article = medline.find("Article")
    if article is None:
        return None

    title_elem = article.find("ArticleTitle")
    title = title_elem.text if title_elem is not None else ""

    abstract_parts = []
    abstract_elem = article.find("Abstract")
    if abstract_elem is not None:
        for text_elem in abstract_elem.findall("AbstractText"):
            label = text_elem.get("Label", "")
            text = text_elem.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    authors = []
    author_list = article.find("AuthorList")
    if author_list is not None:
        for author in author_list.findall("Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None:
                name = last.text
                if fore is not None:
                    name = f"{fore.text} {name}"
                authors.append(name)

    journal_elem = article.find(".//Journal/Title")
    journal = journal_elem.text if journal_elem is not None else ""

    year = ""
    date_elem = article.find(".//PubDate/Year")
    if date_elem is not None:
        year = date_elem.text
    else:
        medline_date = article.find(".//PubDate/MedlineDate")
        if medline_date is not None and medline_date.text:
            year = medline_date.text[:4]

    doi = ""
    for id_elem in article_elem.findall(".//ArticleIdList/ArticleId"):
        if id_elem.get("IdType") == "doi":
            doi = id_elem.text or ""
            break

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors[:5],
        "journal": journal,
        "year": year,
        "doi": doi,
        "abstract": abstract[:1500],
        "source": "pubmed",
    }


def _pubmed_esearch_count(query: str) -> int:
    """Execute a PubMed eSearch and return the result count."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = _ncbi_params({"db": "pubmed", "term": query, "retmode": "json", "retmax": "0"})
    url = f"{search_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        data = json.loads(request_with_retry(req, timeout=15).decode())
        return int(data.get("esearchresult", {}).get("count", 0))
    except Exception:
        return 0


@tool(
    description=(
        "Search PubMed for scientific articles with semantic re-ranking. "
        "Over-fetches 3x, then re-ranks by semantic similarity. "
        "Supports PubMed query syntax, MeSH terms, and field tags."
    ),
    category="literature",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": "PubMed search query (supports MeSH terms, field tags like [Title/Abstract]).",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 10).",
            "default": 10,
        },
        "rerank_query": {
            "type": "string",
            "description": "Optional richer intent string for semantic re-ranking. Uses search query if not provided.",
            "default": "",
        },
    },
)
def search_pubmed(
    query: str,
    max_results: int = 10,
    rerank_query: str = "",
) -> dict:
    """Search PubMed for scientific articles with semantic re-ranking."""
    fetch_count = max_results * config.OVERFETCH_RATIO

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = _ncbi_params({
        "db": "pubmed",
        "term": query,
        "retmax": str(fetch_count),
        "sort": "relevance",
        "retmode": "json",
    })
    url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        data = json.loads(request_with_retry(req, timeout=15).decode())
    except Exception as e:
        return {
            "status": "error",
            "summary": f"PubMed search failed: {e}",
            "data": [],
            "suggestions": ["Try simpler query terms or check network connectivity."],
        }

    pmids = data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return {
            "status": "partial",
            "summary": f"No PubMed results for: {query[:60]}",
            "data": [],
            "count": 0,
            "suggestions": [
                "Try broader terms or remove restrictive field tags.",
                "Search OpenAlex for different coverage.",
            ],
        }

    # Fetch article details
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = _ncbi_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    })
    url = f"{fetch_url}?{urllib.parse.urlencode(fetch_params)}"
    req = urllib.request.Request(url)

    try:
        xml_data = request_with_retry(req, timeout=30).decode()
    except Exception as e:
        return {
            "status": "error",
            "summary": f"PubMed fetch failed: {e}",
            "data": [],
        }

    articles = []
    try:
        root = ET.fromstring(xml_data)
        for article_elem in root.findall(".//PubmedArticle"):
            article = _parse_pubmed_article(article_elem)
            if article:
                articles.append(article)
    except ET.ParseError as e:
        return {
            "status": "error",
            "summary": f"Failed to parse PubMed XML: {e}",
            "data": [],
        }

    if not articles:
        return {
            "status": "partial",
            "summary": "PubMed returned results but could not parse any articles",
            "data": [],
            "count": 0,
        }

    # Semantic re-ranking
    reranked = _semantic_rerank(
        query=rerank_query or query,
        papers=articles,
        max_results=max_results,
    )

    return {
        "status": "success",
        "summary": f"Found {len(reranked)} PubMed articles for: {query[:60]}",
        "data": reranked,
        "count": len(reranked),
    }


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def _reconstruct_openalex_abstract(work: dict, max_chars: int = 1500) -> str:
    """Reconstruct abstract text from OpenAlex inverted index."""
    inv_index = work.get("abstract_inverted_index")
    if not inv_index:
        return ""
    word_positions = []
    for word, positions in inv_index.items():
        for pos in positions:
            try:
                word_positions.append((int(pos), word))
            except (ValueError, TypeError):
                continue
    word_positions.sort()
    return " ".join(w for _, w in word_positions)[:max_chars]


def _clean_openalex_doi(work: dict) -> str:
    """Extract clean DOI string from an OpenAlex work dict."""
    doi = work.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    return doi


@tool(
    description=(
        "Search OpenAlex for scholarly works with semantic re-ranking. "
        "Covers broader literature than PubMed (preprints, conference papers). "
        "Results include citation counts for gauging paper impact."
    ),
    category="literature",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": "Search query string.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 20).",
            "default": 20,
        },
        "sort_by_citations": {
            "type": "boolean",
            "description": "If true, sort by citation count; otherwise by publication year.",
            "default": True,
        },
        "rerank_query": {
            "type": "string",
            "description": "Optional richer intent for semantic re-ranking.",
            "default": "",
        },
    },
)
def search_openalex(
    query: str,
    max_results: int = 20,
    sort_by_citations: bool = True,
    rerank_query: str = "",
) -> dict:
    """Search OpenAlex for scholarly works with semantic re-ranking."""
    fetch_count = min(max_results * config.OVERFETCH_RATIO, 200)

    sort_param = "cited_by_count:desc" if sort_by_citations else "publication_year:desc"
    params = {
        "search": query,
        "per-page": str(fetch_count),
        "mailto": config.OPENALEX_EMAIL,
        "sort": sort_param,
    }
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY
    url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        data = json.loads(request_with_retry(req, timeout=30).decode())
    except Exception as e:
        return {
            "status": "error",
            "summary": f"OpenAlex search failed: {e}",
            "data": [],
            "suggestions": ["Try different query terms."],
        }

    results = []
    for work in data.get("results", []):
        abstract = _reconstruct_openalex_abstract(work)
        authors = []
        for authorship in work.get("authorships", [])[:5]:
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        results.append({
            "title": work.get("title", ""),
            "year": work.get("publication_year"),
            "doi": _clean_openalex_doi(work),
            "cited_by_count": work.get("cited_by_count", 0),
            "authors": authors,
            "abstract": abstract,
            "openalex_id": work.get("id", ""),
            "source": "openalex",
        })

    if not results:
        return {
            "status": "partial",
            "summary": f"No OpenAlex results for: {query[:60]}",
            "data": [],
            "count": 0,
            "suggestions": ["Try broader search terms or check PubMed."],
        }

    reranked = _semantic_rerank(
        query=rerank_query or query,
        papers=results,
        max_results=max_results,
    )

    return {
        "status": "success",
        "summary": f"Found {len(reranked)} OpenAlex works for: {query[:60]}",
        "data": reranked,
        "count": len(reranked),
    }


# ---------------------------------------------------------------------------
# Europe PMC
# ---------------------------------------------------------------------------

def _europepmc_oa_pdf_url(result: dict) -> str:
    """Pick an 'Open access' PDF URL from a Europe PMC result's fullTextUrlList.

    The ``isOpenAccess`` boolean is known to lag the real license, so the
    fullTextUrlList entry is the more trustworthy availability signal.
    """
    urls = (result.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    for entry in urls:
        if (entry.get("availability") == "Open access"
                and entry.get("documentStyle") == "pdf"):
            return entry.get("url", "")
    pmcid = result.get("pmcid")
    if pmcid:
        # The render endpoint is the reliable PDF route for OA-subset papers.
        return f"https://europepmc.org/articles/{pmcid}?pdf=render"
    return ""


@tool(
    description=(
        "Search Europe PMC for biomedical/life-science articles with semantic "
        "re-ranking. Best coverage of the biochem/molbio journal literature and, "
        "unlike PubMed, returns PMCID + open-access flags so full text can be "
        "fetched later. Supports free-text, TITLE:\"...\", DOI:..., EXT_ID:<pmid>."
    ),
    category="literature",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": "Europe PMC query (free-text, or fielded e.g. TITLE:\"...\", AUTH:\"...\", DOI:...).",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 20).",
            "default": 20,
        },
        "rerank_query": {
            "type": "string",
            "description": "Optional richer intent for semantic re-ranking.",
            "default": "",
        },
    },
)
def search_europepmc(
    query: str,
    max_results: int = 20,
    rerank_query: str = "",
) -> dict:
    """Search Europe PMC for scientific articles with semantic re-ranking."""
    fetch_count = min(max_results * config.OVERFETCH_RATIO, 100)
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",  # includes abstractText + fullTextUrlList
        "pageSize": str(fetch_count),
        "email": config.CONTACT_EMAIL,
    }
    url = f"{config.EUROPEPMC_API}/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        data = json.loads(request_with_retry(req, timeout=30).decode())
    except Exception as e:
        return {
            "status": "error",
            "summary": f"Europe PMC search failed: {e}",
            "data": [],
            "suggestions": ["Try simpler query terms or check network connectivity."],
        }

    results = []
    for r in data.get("resultList", {}).get("result", []):
        author_string = r.get("authorString", "") or ""
        authors = [a.strip() for a in author_string.split(",") if a.strip()][:5]
        results.append({
            "title": r.get("title", ""),
            "year": int(r["pubYear"]) if str(r.get("pubYear", "")).isdigit() else None,
            "doi": r.get("doi", "") or "",
            "pmid": r.get("pmid", "") or "",
            "pmcid": r.get("pmcid", "") or "",
            "journal": r.get("journalTitle", "") or "",
            "authors": authors,
            "abstract": (r.get("abstractText", "") or "")[:1500],
            "cited_by_count": r.get("citedByCount", 0),
            "is_open_access": (r.get("isOpenAccess") == "Y"),
            "oa_pdf_url": _europepmc_oa_pdf_url(r),
            "source": "europepmc",
        })

    if not results:
        return {
            "status": "partial",
            "summary": f"No Europe PMC results for: {query[:60]}",
            "data": [],
            "count": 0,
            "suggestions": ["Try broader terms, or search PubMed/OpenAlex for other coverage."],
        }

    reranked = _semantic_rerank(query=rerank_query or query, papers=results,
                                max_results=max_results)
    return {
        "status": "success",
        "summary": f"Found {len(reranked)} Europe PMC articles for: {query[:60]}",
        "data": reranked,
        "count": len(reranked),
    }


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

_S2_FIELDS = "title,year,venue,abstract,externalIds,openAccessPdf,citationCount,authors"


@tool(
    description=(
        "Search Semantic Scholar for scholarly works with semantic re-ranking. "
        "Uniquely returns the full identifier set (DOI, PMID, PMCID, arXiv id) "
        "AND a direct open-access PDF link in a single call, so it is the best "
        "front door for feeding downstream full-text retrieval."
    ),
    category="literature",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": "Free-text search query.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 20).",
            "default": 20,
        },
        "rerank_query": {
            "type": "string",
            "description": "Optional richer intent for semantic re-ranking.",
            "default": "",
        },
    },
)
def search_semantic_scholar(
    query: str,
    max_results: int = 20,
    rerank_query: str = "",
) -> dict:
    """Search Semantic Scholar for scholarly works with semantic re-ranking."""
    fetch_count = min(max_results * config.OVERFETCH_RATIO, 100)
    params = {
        "query": query,
        "limit": str(fetch_count),
        "fields": _S2_FIELDS,
    }
    url = f"{config.SEMANTIC_SCHOLAR_API}/paper/search?{urllib.parse.urlencode(params)}"
    headers = {"Accept": "application/json"}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY
    req = urllib.request.Request(url, headers=headers)

    try:
        data = json.loads(request_with_retry(req, timeout=30).decode())
    except Exception as e:
        # The keyless pool 429s aggressively; surface an actionable hint.
        rate_limited = "429" in str(e)
        return {
            "status": "partial" if rate_limited else "error",
            "summary": f"Semantic Scholar search failed: {e}",
            "data": [],
            "suggestions": (
                ["Rate limited on the keyless pool. Set ENZYME_AGENT_S2_API_KEY, "
                 "or fall back to search_europepmc / search_crossref."]
                if rate_limited else
                ["Try simpler query terms or a different literature source."]
            ),
        }

    results = []
    for p in data.get("data", []) or []:
        ext = p.get("externalIds") or {}
        oa = p.get("openAccessPdf") or {}
        authors = [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")][:5]
        results.append({
            "title": p.get("title", "") or "",
            "year": p.get("year"),
            "doi": ext.get("DOI", "") or "",
            "pmid": ext.get("PubMed", "") or "",
            "pmcid": ext.get("PubMedCentral", "") or "",
            "arxiv_id": ext.get("ArXiv", "") or "",
            "journal": p.get("venue", "") or "",
            "authors": authors,
            "abstract": (p.get("abstract") or "")[:1500],
            "cited_by_count": p.get("citationCount", 0),
            "oa_status": oa.get("status", "") or "",
            "oa_pdf_url": oa.get("url", "") or "",
            "source": "semantic_scholar",
        })

    if not results:
        return {
            "status": "partial",
            "summary": f"No Semantic Scholar results for: {query[:60]}",
            "data": [],
            "count": 0,
            "suggestions": ["Try broader terms or another literature source."],
        }

    reranked = _semantic_rerank(query=rerank_query or query, papers=results,
                                max_results=max_results)
    return {
        "status": "success",
        "summary": f"Found {len(reranked)} Semantic Scholar works for: {query[:60]}",
        "data": reranked,
        "count": len(reranked),
    }


# ---------------------------------------------------------------------------
# Crossref (title -> DOI resolver)
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    """Strip JATS/HTML tags (Crossref abstracts are wrapped in JATS)."""
    return _TAG_RE.sub("", text or "").strip()


@tool(
    description=(
        "Resolve a title or bibliographic string to a DOI (and rich metadata) "
        "via Crossref. Best title->DOI precision; inspect the returned "
        "'crossref_score' to judge match confidence. Metadata only (no full text) "
        "- use it to obtain clean DOIs to hand to full-text retrieval, and to "
        "reach preprint servers (bioRxiv/ChemRxiv) that lack keyword search."
    ),
    category="literature",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": "Bibliographic query (title works best; keywords also accepted).",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 10).",
            "default": 10,
        },
        "rerank_query": {
            "type": "string",
            "description": "Optional richer intent for semantic re-ranking.",
            "default": "",
        },
    },
)
def search_crossref(
    query: str,
    max_results: int = 10,
    rerank_query: str = "",
) -> dict:
    """Resolve bibliographic queries to DOIs + metadata via Crossref."""
    fetch_count = min(max_results * config.OVERFETCH_RATIO, 50)
    params = {
        "query.bibliographic": query,
        "rows": str(fetch_count),
        "mailto": config.CONTACT_EMAIL,
        "select": ("DOI,title,container-title,author,issued,"
                   "is-referenced-by-count,score,abstract,type"),
    }
    url = f"{config.CROSSREF_API}/works?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        data = json.loads(request_with_retry(req, timeout=30).decode())
    except Exception as e:
        return {
            "status": "error",
            "summary": f"Crossref search failed: {e}",
            "data": [],
            "suggestions": ["Try a more complete title string."],
        }

    results = []
    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title") or [""]
        journal_list = item.get("container-title") or [""]
        authors = []
        for a in item.get("author", [])[:5]:
            name = " ".join(x for x in (a.get("given", ""), a.get("family", "")) if x).strip()
            if name:
                authors.append(name)
        year = None
        date_parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        if date_parts and isinstance(date_parts[0], int):
            year = date_parts[0]
        results.append({
            "title": title_list[0] if title_list else "",
            "year": year,
            "doi": item.get("DOI", "") or "",
            "journal": journal_list[0] if journal_list else "",
            "authors": authors,
            "abstract": _strip_tags(item.get("abstract", ""))[:1500],
            "cited_by_count": item.get("is-referenced-by-count", 0),
            "crossref_score": round(float(item.get("score", 0.0)), 2),
            "type": item.get("type", ""),
            "source": "crossref",
        })

    if not results:
        return {
            "status": "partial",
            "summary": f"No Crossref results for: {query[:60]}",
            "data": [],
            "count": 0,
            "suggestions": ["Provide a more complete or exact title."],
        }

    reranked = _semantic_rerank(query=rerank_query or query, papers=results,
                                max_results=max_results)
    return {
        "status": "success",
        "summary": f"Found {len(reranked)} Crossref records for: {query[:60]}",
        "data": reranked,
        "count": len(reranked),
    }


# ---------------------------------------------------------------------------
# Entity extraction from paper abstracts
# ---------------------------------------------------------------------------

_GENE_RE = re.compile(r'\b([A-Z][a-z]{1,3}[A-Z]\d{0,2})\b')
_GENE_RE2 = re.compile(r'\b([A-Z]{2,4}[a-z][A-Z0-9])\b')
_EC_RE = re.compile(r'EC\s*(\d+\.\d+\.\d+\.[\d\*\-]+)')
_GENBANK_RE = re.compile(r'\b([A-Z]{3}\d{5}(?:\.\d+)?)\b')
_GENBANK_PROTEIN_RE = re.compile(r'\b([A-Z]{3}\d{5,7}(?:\.\d+)?)\b')
_FAMILY_RE = re.compile(
    r'((?:[A-Za-z][\w-]*\s+){0,2}[A-Za-z][\w-]*)\s+'
    r'(synthetase|ligase|hydrolase|transferase|oxidase|reductase|'
    r'dehydrogenase|kinase|isomerase|lyase|synthase|amidase|esterase|'
    r'peptidase|protease)\b',
    re.IGNORECASE,
)

_GENE_NAME_BLACKLIST = frozenset({
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "UTP", "UDP", "UMP",
    "CTP", "CDP", "CMP", "NAD", "FAD", "FMN", "CoA", "PPi",
    "DNA", "RNA", "mRNA", "tRNA", "rRNA", "cDNA", "sRNA", "dsDNA",
    "PCR", "NMR", "SDS", "PAGE", "HPLC", "FPLC", "EDTA", "IPTG",
    "DMSO", "HEPES", "MOPS",
    "NRPS", "BLAST", "HMM", "MSA", "PDB", "NCBI", "KEGG",
    "Fig", "Tab", "Ref", "Sup", "Oct", "Nov", "Dec", "Jan",
    "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
    "kDa", "mM", "nM", "pH", "Tris", "His", "Glu", "Asp",
    "Ala", "Val", "Leu", "Ile", "Pro", "Phe", "Trp", "Met",
    "Ser", "Thr", "Cys", "Tyr", "Asn", "Gln", "Lys", "Arg",
    "MeOH", "EtOH",
})


@tool(
    description=(
        "Extract gene names, EC numbers, and enzyme families from a list of "
        "papers. Scans titles and abstracts using regex patterns. Use this after "
        "search_pubmed or search_openalex to discover specific enzyme/gene names "
        "that you wouldn't know to search for directly."
    ),
    category="literature",
    parallel_safe=True,
    injected_params=["run_dir"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "papers": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Optional paper dicts. If omitted, uses literature artifacts already collected in this run.",
            "default": [],
        },
    },
)
def extract_entities(
    papers: list[dict] | None = None,
    run_dir: str | None = None,
) -> dict:
    """Extract gene names, EC numbers, enzyme families, and GenBank accessions from papers."""
    from collections import defaultdict

    if papers is None:
        papers = []
    if not papers and run_dir:
        # Reuse the durable search artifacts already written by this run rather
        # than making the model reconstruct a large paper payload as an argument.
        papers = _recover_literature_papers(run_dir)

    gene_hits = defaultdict(set)
    ec_hits = defaultdict(set)
    family_hits = defaultdict(set)
    genbank_hits = defaultdict(set)

    for paper in papers:
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        text = f"{title} {abstract}"
        paper_title = title[:120] if title else "(untitled)"

        for m in _GENE_RE.finditer(text):
            name = m.group(1)
            if name not in _GENE_NAME_BLACKLIST:
                gene_hits[name].add(paper_title)
        for m in _GENE_RE2.finditer(text):
            name = m.group(1)
            if name not in _GENE_NAME_BLACKLIST:
                gene_hits[name].add(paper_title)

        for m in _EC_RE.finditer(text):
            ec_hits[m.group(1)].add(paper_title)

        for m in _FAMILY_RE.finditer(text):
            prefix = m.group(1).strip()
            suffix = m.group(2).lower()
            if len(prefix) < 2 or prefix.lower() in {"the", "a", "an", "this", "each"}:
                continue
            family_hits[f"{prefix} {suffix}"].add(paper_title)

        for m in _GENBANK_RE.finditer(text):
            acc = m.group(1)
            if acc not in _GENE_NAME_BLACKLIST:
                genbank_hits[acc].add(paper_title)
        for m in _GENBANK_PROTEIN_RE.finditer(text):
            acc = m.group(1)
            if acc not in _GENE_NAME_BLACKLIST and len(acc) >= 8:
                genbank_hits[acc].add(paper_title)

    def _build(hits, key_name):
        items = []
        for name, titles in hits.items():
            items.append({
                key_name: name,
                "count": len(titles),
                "papers": sorted(titles)[:5],
            })
        items.sort(key=lambda x: x["count"], reverse=True)
        return items

    entities = {
        "gene_names": _build(gene_hits, "name"),
        "ec_numbers": _build(ec_hits, "ec"),
        "enzyme_families": _build(family_hits, "name"),
        "genbank_accessions": _build(genbank_hits, "accession"),
    }

    return {
        "status": "success",
        "summary": (
            f"Extracted {len(entities['gene_names'])} gene names, "
            f"{len(entities['ec_numbers'])} EC numbers, "
            f"{len(entities['enzyme_families'])} enzyme families, "
            f"{len(entities['genbank_accessions'])} GenBank accessions"
        ),
        "data": entities,
    }


# ---------------------------------------------------------------------------
# Deep (multi-hop snowball) literature search
# ---------------------------------------------------------------------------

def _batch_fetch_openalex_works(openalex_ids: list[str]) -> list[dict]:
    """Batch-fetch OpenAlex works by ID, returning paper dicts with abstracts."""
    chunk_size = 50
    chunks = [openalex_ids[i:i + chunk_size]
              for i in range(0, len(openalex_ids), chunk_size)]

    def _fetch_chunk(chunk):
        pipe_ids = "|".join(chunk)
        url = (
            f"https://api.openalex.org/works?"
            f"filter=openalex:{pipe_ids}"
            f"&per-page={len(chunk)}"
            f"&mailto={config.OPENALEX_EMAIL}"
            f"{_openalex_auth_qs()}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            data = json.loads(request_with_retry(req, timeout=30).decode())
        except Exception as e:
            logger.warning("OpenAlex batch fetch failed: %s", e)
            return []
        results = []
        for work in data.get("results", []):
            results.append({
                "title": work.get("title", ""),
                "year": work.get("publication_year"),
                "doi": _clean_openalex_doi(work),
                "cited_by_count": work.get("cited_by_count", 0),
                "abstract": _reconstruct_openalex_abstract(work),
                "openalex_id": work.get("id", ""),
            })
        return results

    if len(chunks) <= 1:
        return _fetch_chunk(chunks[0]) if chunks else []

    all_results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_fetch_chunk, c) for c in chunks]
        for f in futures:
            all_results.extend(f.result())
    return all_results


def _openalex_work_to_paper(work: dict) -> dict:
    """Parse a raw OpenAlex work object into the standard paper dict."""
    authors = []
    for authorship in work.get("authorships", [])[:5]:
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)
    return {
        "title": work.get("title", "") or "",
        "year": work.get("publication_year"),
        "doi": _clean_openalex_doi(work),
        "cited_by_count": work.get("cited_by_count", 0),
        "authors": authors,
        "abstract": _reconstruct_openalex_abstract(work),
        "openalex_id": work.get("id", ""),
        "source": "openalex",
    }


def _fetch_forward_citations(cited_by_urls: list[str], per_seed: int,
                             total_cap: int) -> list[dict]:
    """Fetch the most-cited works that CITE each seed (forward snowball).

    A seed's ``cited_by_api_url`` can point at thousands of citers, so take only
    the top ``per_seed`` by citation count per seed, and stop at ``total_cap``.
    """
    papers: list[dict] = []
    for cb in cited_by_urls:
        if len(papers) >= total_cap:
            break
        sep = "&" if "?" in cb else "?"
        url = (f"{cb}{sep}per-page={per_seed}&sort=cited_by_count:desc"
               f"&mailto={config.OPENALEX_EMAIL}{_openalex_auth_qs()}")
        try:
            data = json.loads(request_with_retry(
                urllib.request.Request(url, headers={"Accept": "application/json"}),
                timeout=30).decode())
        except Exception as e:
            logger.warning("Forward-citation fetch failed: %s", e)
            continue
        for work in data.get("results", []):
            papers.append(_openalex_work_to_paper(work))
            if len(papers) >= total_cap:
                break
    return papers


def _dedup_papers(papers: list[dict]) -> list[dict]:
    """Dedup a list of paper dicts across sources.

    Keyed by DOI first, then PMID, then OpenAlex id, then title. DOI/PMID lead so
    the same paper returned by different citation graphs (OpenAlex work, Semantic
    Scholar record, Europe PMC/PubMed record) collapses to one entry even when the
    sources expose different id subsets.
    """
    seen, out = set(), []
    for p in papers:
        key = ((p.get("doi") or "").strip().lower()
               or (p.get("pmid") or "").strip()
               or (p.get("openalex_id") or "").strip()
               or (p.get("title") or "").strip().lower())
        if not key:
            out.append(p)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _round_robin(lists: list[list]):
    """Interleave any number of lists for source diversity."""
    i, maxlen = 0, max((len(x) for x in lists), default=0)
    while i < maxlen:
        for lst in lists:
            if i < len(lst):
                yield lst[i]
        i += 1


def _interleave(list_a: list, list_b: list):
    """Interleave two lists for diversity."""
    i, j = 0, 0
    while i < len(list_a) or j < len(list_b):
        if i < len(list_a):
            yield list_a[i]
            i += 1
        if j < len(list_b):
            yield list_b[j]
            j += 1


@tool(
    description=(
        "Multi-hop snowball literature search. Finds seed papers across PubMed, "
        "OpenAlex, Europe PMC and Semantic Scholar, then follows their citation "
        "networks in BOTH directions — backward (references) AND forward (papers "
        "that cite the seed) — and extracts gene names and EC numbers from all "
        "collected abstracts. The single best tool for discovering specific "
        "enzyme/gene names you wouldn't reach by keyword search: forward citations "
        "surface newer work building on a mechanistically-relevant seed."
    ),
    category="literature",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": "Search query for finding seed papers.",
        },
        "max_seed_papers": {
            "type": "integer",
            "description": "Number of top seed papers to follow (default 5).",
            "default": 5,
        },
        "max_refs_per_seed": {
            "type": "integer",
            "description": "Max references to follow per seed paper (default 15).",
            "default": 15,
        },
        "rerank_query": {
            "type": "string",
            "description": "Optional richer intent for seed re-ranking.",
            "default": "",
        },
    },
)
def deep_literature_search(
    query: str,
    max_seed_papers: int = 5,
    max_refs_per_seed: int = 15,
    rerank_query: str = "",
) -> dict:
    """Multi-hop snowball search with automatic entity extraction."""
    seed_pool_size = max(max_seed_papers * 4, 20)

    # Step 1: Search all sources in parallel for seed papers.
    seed_fns = (search_pubmed, search_openalex, search_europepmc,
                search_semantic_scholar)
    with ThreadPoolExecutor(max_workers=len(seed_fns)) as pool:
        futs = [pool.submit(fn, query, max_results=seed_pool_size,
                            rerank_query=rerank_query) for fn in seed_fns]
        source_papers = []
        for fut in futs:
            r = fut.result()
            source_papers.append(
                r.get("data", []) if r.get("status") != "error" else [])

    # Step 2: Deduplicate by DOI, round-robin across sources for diversity.
    seen_dois = set()
    seeds = []
    for paper in _round_robin(source_papers):
        if isinstance(paper, dict) and "error" in paper:
            continue
        doi = (paper.get("doi") or "").strip().lower()
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        seeds.append(paper)
        if len(seeds) >= max_seed_papers:
            break

    if not seeds:
        return {
            "status": "partial",
            "summary": f"No seed papers found for: {query[:60]}",
            "data": {"seed_papers": [], "extracted_entities": {}, "stats": {}},
        }

    # Step 3: From each seed's OpenAlex work, collect BACKWARD reference ids and
    # the FORWARD cited_by URL. The global budget is split so forward citations
    # do not starve backward coverage (a landmark seed has thousands of citers).
    BACKWARD_CAP, FORWARD_CAP = 60, 40
    seed_dois = [s["doi"] for s in seeds if s.get("doi")]
    backward_ids: list[str] = []
    cited_by_urls: list[str] = []

    if seed_dois:
        doi_filter = "|".join(f"https://doi.org/{d}" for d in seed_dois)
        url = (
            f"https://api.openalex.org/works?"
            f"filter=doi:{urllib.parse.quote(doi_filter, safe='|:/')}"
            f"&per-page={len(seed_dois)}"
            f"&mailto={config.OPENALEX_EMAIL}"
            f"{_openalex_auth_qs()}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            data = json.loads(request_with_retry(req, timeout=30).decode())
            for work in data.get("results", []):
                backward_ids.extend(
                    work.get("referenced_works", [])[:max_refs_per_seed])
                # OpenAlex dropped cited_by_api_url from work objects, so build
                # the forward (cites:) query from the work's own id.
                wid = (work.get("id") or "").rsplit("/", 1)[-1]  # W123...
                if wid and work.get("cited_by_count", 0) > 0:
                    cited_by_urls.append(
                        f"{config.OPENALEX_API}/works?filter=cites:{wid}")
        except Exception as e:
            logger.warning("Deep search: failed to fetch seed works: %s", e)

    # Step 4: fetch backward references (batch by id) and forward citations
    # (top-cited citers per seed), each within its budget, then dedup together.
    unique_back = list(dict.fromkeys(backward_ids))[:BACKWARD_CAP]
    backward_papers = _batch_fetch_openalex_works(unique_back) if unique_back else []
    forward_papers = _fetch_forward_citations(
        cited_by_urls, per_seed=max_refs_per_seed, total_cap=FORWARD_CAP)
    ref_papers = _dedup_papers(backward_papers + forward_papers)

    # Step 5: Extract entities from all collected papers
    all_papers = seeds + ref_papers
    papers_with_text = [p for p in all_papers
                        if (p.get("abstract") or "") or (p.get("title") or "")]

    entities_result = extract_entities(papers_with_text)
    entities = entities_result.get("data", {})

    seed_summaries = [{
        "title": s.get("title", ""),
        "year": s.get("year"),
        "doi": s.get("doi", ""),
        "abstract": (s.get("abstract") or "")[:300],
    } for s in seeds]

    return {
        "status": "success",
        "summary": (
            f"Deep search: {len(seeds)} seeds, {len(backward_papers)} backward + "
            f"{len(forward_papers)} forward citations, "
            f"{len(entities.get('gene_names', []))} genes, "
            f"{len(entities.get('ec_numbers', []))} ECs extracted"
        ),
        "data": {
            "seed_papers": seed_summaries,
            "extracted_entities": entities,
            "stats": {
                "seed_count": len(seeds),
                "backward_refs": len(backward_papers),
                "forward_citations": len(forward_papers),
                "references_fetched": len(ref_papers),
                "total_papers_scanned": len(papers_with_text),
            },
        },
    }


# ---------------------------------------------------------------------------
# Sequence-based literature search (PaperBLAST alternative)
# ---------------------------------------------------------------------------

def _fetch_uniprot_json(accession: str) -> dict | None:
    """Fetch a UniProt entry as JSON. Returns None on failure."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        return json.loads(request_with_retry(req, timeout=20).decode())
    except Exception as exc:
        logger.warning("UniProt JSON fetch failed for %s: %s", accession, exc)
        return None


def _extract_publications_from_uniprot(entry: dict) -> list[dict]:
    """Extract publication data from a UniProt JSON entry."""
    pubs = []
    for ref in entry.get("references", []):
        cit = ref.get("citation", {})
        title = cit.get("title", "")
        if not title:
            continue
        journal = cit.get("journal", "")
        year = cit.get("publicationDate", "")[:4] if cit.get("publicationDate") else ""
        pmid = ""
        doi = ""
        for xref in cit.get("citationCrossReferences", []):
            if xref.get("database") == "PubMed":
                pmid = xref.get("id", "")
            elif xref.get("database") == "DOI":
                doi = xref.get("id", "")
        pubs.append({
            "title": title,
            "journal": journal,
            "year": year,
            "pmid": pmid,
            "doi": doi,
            "abstract": "",
        })
    return pubs


def _get_interpro_type(ipr_id: str) -> str:
    """Query the InterPro API for an entry's type (family, domain, etc.)."""
    url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{ipr_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        data = json.loads(request_with_retry(req, timeout=10).decode())
        return data.get("metadata", {}).get("type", "")
    except Exception:
        return ""


def _find_interpro_family(entry: dict) -> tuple[str, str]:
    """Extract the most specific InterPro family entry from a UniProt entry.

    Queries the InterPro API to determine entry types, preferring entries
    of type 'family' over 'domain' or 'homologous_superfamily'.
    """
    ipr_entries = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "InterPro":
            continue
        ipr_id = xref.get("id", "")
        props = {p["key"]: p["value"] for p in xref.get("properties", [])}
        name = props.get("EntryName", "")
        ipr_entries.append((ipr_id, name))

    if not ipr_entries:
        return "", ""

    for ipr_id, name in ipr_entries:
        entry_type = _get_interpro_type(ipr_id)
        if entry_type == "family":
            return ipr_id, name

    return ipr_entries[0][0], ipr_entries[0][1]


def _search_uniprot_family_members(
    family_id: str, max_members: int = 20
) -> list[dict]:
    """Search UniProt for reviewed members of an InterPro family."""
    query = f"(xref:interpro-{family_id}) AND (reviewed:true)"
    fields = "accession,gene_names,organism_name,protein_name,lit_pubmed_id"
    params = urllib.parse.urlencode({
        "query": query,
        "fields": fields,
        "size": str(max_members),
        "format": "json",
    })
    url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        data = json.loads(request_with_retry(req, timeout=30).decode())
        members = []
        for r in data.get("results", []):
            acc = r.get("primaryAccession", "")
            genes = []
            for g in r.get("genes", []):
                gn = g.get("geneName", {}).get("value", "")
                if gn:
                    genes.append(gn)
            org = r.get("organism", {}).get("scientificName", "")
            name = ""
            pd = r.get("proteinDescription", {})
            rec_name = pd.get("recommendedName", {})
            if rec_name:
                name = rec_name.get("fullName", {}).get("value", "")
            members.append({
                "accession": acc,
                "gene_names": genes,
                "organism": org,
                "protein_name": name,
            })
        return members
    except Exception as exc:
        logger.warning("UniProt family search failed for %s: %s", family_id, exc)
        return []


def _search_pubmed_by_gene(gene_name: str, max_results: int = 5) -> list[dict]:
    """Quick PubMed search for a gene name, returning minimal paper dicts."""
    query = f'"{gene_name}" AND enzyme'
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = urllib.parse.urlencode(_ncbi_params({
        "db": "pubmed", "term": query,
        "retmax": str(max_results), "sort": "relevance", "retmode": "json",
    }))
    req = urllib.request.Request(
        f"{search_url}?{params}", headers={"Accept": "application/json"}
    )
    try:
        data = json.loads(request_with_retry(req, timeout=15).decode())
    except Exception:
        return []

    pmids = data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = urllib.parse.urlencode(_ncbi_params({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
    }))
    req2 = urllib.request.Request(f"{fetch_url}?{fetch_params}")
    try:
        xml_data = request_with_retry(req2, timeout=15).decode()
        root = ET.fromstring(xml_data)
    except Exception:
        return []

    papers = []
    for article in root.findall(".//PubmedArticle"):
        parsed = _parse_pubmed_article(article)
        if parsed:
            papers.append(parsed)
    return papers


@tool(
    description=(
        "Sequence-based literature search: finds publications about homologous "
        "enzymes in the same InterPro protein family. Given a UniProt accession, "
        "this tool identifies the protein's InterPro family, finds reviewed "
        "family members, collects their curated UniProt publications, and runs "
        "PubMed gene-name searches per homolog. Use this to discover "
        "characterization papers about related enzymes that keyword searches "
        "miss because they use organism-specific gene names."
    ),
    category="literature",
    parallel_safe=True,
    params={
        "accession": {
            "type": "string",
            "description": (
                "UniProt accession of a representative family member "
                "(e.g. a reviewed UniProt accession for the selected family)."
            ),
        },
        "max_homologs": {
            "type": "integer",
            "description": "Maximum homologs to process (default 20).",
            "default": 20,
        },
        "rerank_query": {
            "type": "string",
            "description": (
                "Target reaction description for semantic re-ranking of "
                "collected publications (optional)."
            ),
            "default": "",
        },
    },
)
def sequence_literature_search(
    accession: str,
    max_homologs: int = 20,
    rerank_query: str = "",
) -> dict:
    """Find publications about homologous enzymes via InterPro family lookup."""
    import time

    accession = accession.strip()
    if not accession:
        return {
            "status": "error",
            "summary": "accession is required.",
            "data": None,
            "count": 0,
            "suggestions": ["Provide a UniProt accession (e.g. 'Q81RQ8')."],
        }

    # Step 1: Fetch the seed protein and identify its InterPro family
    seed_entry = _fetch_uniprot_json(accession)
    if seed_entry is None:
        return {
            "status": "error",
            "summary": f"Could not fetch UniProt entry for {accession}.",
            "data": None,
            "count": 0,
            "suggestions": ["Verify the accession ID and try again."],
        }

    family_id, family_name = _find_interpro_family(seed_entry)
    if not family_id:
        return {
            "status": "partial",
            "summary": (
                f"No InterPro family found for {accession}. "
                "Cannot perform family-based homolog search."
            ),
            "data": {"homologs": [], "publications": [], "extracted_entities": {}},
            "count": 0,
            "suggestions": [
                "Try deep_literature_search with the enzyme's gene name instead.",
                "The protein may lack InterPro family annotation.",
            ],
        }

    # Step 2: Find reviewed family members
    members = _search_uniprot_family_members(family_id, max_homologs)
    if not members:
        return {
            "status": "partial",
            "summary": (
                f"InterPro family {family_id} ({family_name}) found for "
                f"{accession}, but no reviewed members retrieved."
            ),
            "data": {"family_id": family_id, "family_name": family_name,
                     "homologs": [], "publications": [], "extracted_entities": {}},
            "count": 0,
            "suggestions": ["Try searching UniProt directly for this family."],
        }

    # Step 3: Collect publications from two sources per homolog
    all_pubs: list[dict] = []
    seen_pmids: set[str] = set()
    seen_titles_lower: set[str] = set()
    homolog_summaries = []

    for member in members:
        acc = member["accession"]
        genes = member.get("gene_names", [])

        member_pubs_count = 0

        # Source A: UniProt curated references
        entry = _fetch_uniprot_json(acc)
        if entry:
            pubs = _extract_publications_from_uniprot(entry)
            for pub in pubs:
                pmid = pub.get("pmid", "")
                title_lower = (pub.get("title") or "").lower().strip()
                if pmid and pmid in seen_pmids:
                    continue
                if not pmid and title_lower in seen_titles_lower:
                    continue
                if pmid:
                    seen_pmids.add(pmid)
                if title_lower:
                    seen_titles_lower.add(title_lower)
                pub["source_accessions"] = [acc]
                all_pubs.append(pub)
                member_pubs_count += 1

        # Source B: PubMed gene-name search (first gene only, capped)
        if genes:
            time.sleep(0.3)
            gene_pubs = _search_pubmed_by_gene(genes[0], max_results=5)
            for pub in gene_pubs:
                pmid = pub.get("pmid", "")
                title_lower = (pub.get("title") or "").lower().strip()
                if pmid and pmid in seen_pmids:
                    continue
                if not pmid and title_lower in seen_titles_lower:
                    continue
                if pmid:
                    seen_pmids.add(pmid)
                if title_lower:
                    seen_titles_lower.add(title_lower)
                pub["source_accessions"] = [acc]
                all_pubs.append(pub)
                member_pubs_count += 1

        homolog_summaries.append({
            "accession": acc,
            "gene": genes[0] if genes else "",
            "organism": member.get("organism", ""),
            "protein_name": member.get("protein_name", ""),
            "n_papers": member_pubs_count,
        })

        time.sleep(0.3)

    # Step 4: Semantic re-rank if query provided
    if rerank_query and all_pubs:
        all_pubs = _semantic_rerank(
            rerank_query, all_pubs, max_results=len(all_pubs),
            min_score=0.0,
        )

    # Step 5: Extract entities from collected papers
    entities_result = extract_entities(all_pubs)
    entities = entities_result.get("data", {})

    pub_summaries = [{
        "title": p.get("title", ""),
        "year": p.get("year", ""),
        "pmid": p.get("pmid", ""),
        "doi": p.get("doi", ""),
        "source_accessions": p.get("source_accessions", []),
        "semantic_score": p.get("semantic_score"),
    } for p in all_pubs]

    return {
        "status": "success",
        "summary": (
            f"Found {len(all_pubs)} publications across {len(members)} "
            f"homologs of {accession} (InterPro family {family_id}: "
            f"{family_name})"
        ),
        "data": {
            "family_id": family_id,
            "family_name": family_name,
            "homologs": homolog_summaries,
            "publications": pub_summaries,
            "extracted_entities": entities,
        },
        "count": len(all_pubs),
        "suggestions": [
            f"Search for newly discovered gene names: "
            f"{', '.join(e['name'] for e in entities.get('gene_names', [])[:10])}"
        ] if entities.get("gene_names") else [],
    }
