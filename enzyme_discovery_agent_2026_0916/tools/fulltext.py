"""Full-text acquisition: download and store PDFs / full-text XML per run.

Given papers already discovered (carrying their identifiers), fetch the actual
full text via a source-aware fallback chain and save it under the run's
``artifacts/fulltext/`` directory, alongside a ``manifest.json``.

Every fetch is VERIFIED (``%PDF`` magic for PDFs, a JATS ``<article>`` root for
XML) before it is stored -- publisher "PDF" links 403 to HTML and DOI landings
return HTML, so a resolver's ``url_for_pdf`` cannot be trusted without checking
the bytes.  Unverified responses are discarded and the chain moves on.

Chain per paper (first verified hit wins):
  arXiv id             -> arxiv.org/pdf/{id}
  bioRxiv DOI 10.1101  -> api.biorxiv.org version -> www.biorxiv.org .full.pdf
  PMCID                -> Europe PMC fullTextXML (preferred) else render PDF
  oa_pdf_url (S2/EPMC) -> try the link directly
  DOI                  -> Unpaywall best green-OA PDF
ChemRxiv is intentionally not attempted (Cloudflare-gated; not fetchable
unattended).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools._http import request_with_retry
from enzyme_discovery_agent.tools.literature import (
    _recover_literature_papers,
    consolidate_pool,
)

logger = logging.getLogger("enzyme_agent")


def _get(url: str, timeout: int = 60) -> bytes | None:
    """GET raw bytes via the shared client (browser UA, throttle, retry, redact).

    Returns the body, or None on any HTTP/network error (so the chain falls
    through to the next source instead of raising).
    """
    if not url:
        return None
    try:
        return request_with_retry(urllib.request.Request(url), timeout=timeout)
    except Exception as e:  # noqa: BLE001 - any failure = try next source
        logger.debug("fulltext GET failed (%s): %s", url[:90], e)
        return None


# ---------------------------------------------------------------------------
# Verification (never store HTML masquerading as a paper)
# ---------------------------------------------------------------------------

def _is_pdf(content: bytes | None) -> bool:
    return bool(content) and content[:5] == b"%PDF-"


def _is_jats(content: bytes | None) -> bool:
    head = content[:4000].lower() if content else b""
    return b"<article" in head or b"<!doctype article" in head


# ---------------------------------------------------------------------------
# Text extraction (XML is free; PDF needs the optional pymupdf extra)
# ---------------------------------------------------------------------------

def _jats_to_text(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    body = root.find(".//body")
    node = body if body is not None else root
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def _pdf_to_text(pdf_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF (optional 'fulltext' extra)
    except Exception:
        return ""  # store the PDF; text extraction just unavailable
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc).strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("PDF text extraction failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Per-source fetchers: return (bytes, "pdf"|"xml") or None
# ---------------------------------------------------------------------------

def _fetch_arxiv(arxiv_id: str):
    b = _get(f"https://arxiv.org/pdf/{arxiv_id}")
    return (b, "pdf") if _is_pdf(b) else None


def _fetch_biorxiv(doi: str):
    ver = None
    meta = _get(f"{config.BIORXIV_API}/details/biorxiv/"
                f"{urllib.parse.quote(doi, safe='')}", timeout=25)
    if meta:
        try:
            coll = json.loads(meta).get("collection", [])
            if coll:
                ver = coll[-1].get("version")
        except (json.JSONDecodeError, AttributeError, IndexError):
            pass
    suffix = f"v{ver}" if ver else "v1"
    b = _get(f"https://www.biorxiv.org/content/{doi}{suffix}.full.pdf")
    return (b, "pdf") if _is_pdf(b) else None


def _fetch_europepmc(pmcid: str):
    xml = _get(f"{config.EUROPEPMC_API}/{pmcid}/fullTextXML")
    if _is_jats(xml):
        return (xml, "xml")  # preferred: parseable full text
    pdf = _get(f"https://europepmc.org/articles/{pmcid}?pdf=render")
    return (pdf, "pdf") if _is_pdf(pdf) else None


def _fetch_direct_url(url: str):
    b = _get(url)
    if _is_pdf(b):
        return (b, "pdf")
    if _is_jats(b):
        return (b, "xml")
    return None


def _fetch_unpaywall(doi: str):
    meta = _get(f"{config.UNPAYWALL_API}/{urllib.parse.quote(doi, safe='')}"
                f"?email={config.CONTACT_EMAIL}", timeout=25)
    if not meta:
        return None
    try:
        data = json.loads(meta)
    except json.JSONDecodeError:
        return None
    locs = list(data.get("oa_locations") or [])
    # Prefer repository (green) copies over publisher links, which bots-403.
    locs.sort(key=lambda loc: 0 if loc.get("host_type") == "repository" else 1)
    for loc in locs:
        got = _fetch_direct_url(loc.get("url_for_pdf") or "")
        if got:
            return got
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _select_harvest_papers(ranked: list[dict], done: set, cap: int) -> list[dict]:
    """Top-``cap`` not-yet-fetched papers, in relevance-ranked order (pure).

    ``ranked`` is already ordered most-relevant-first, so this keeps the most
    relevant unfetched papers and bounds how many enter the expensive harvest
    (one LLM relevance call + download + extraction each)."""
    todo: list[dict] = []
    for paper in ranked:
        if len(todo) >= cap:
            break
        key = _fulltext_key(paper)
        if key is not None and key not in done:
            todo.append(paper)
    return todo


def _fulltext_key(paper: dict) -> str | None:
    for k in ("doi", "pmcid", "arxiv_id", "pmid"):
        v = (str(paper.get(k) or "")).strip()
        if v:
            return v.lower()
    return None


def _slug(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", key)[:120]


def _paper_basename(paper: dict, key: str, dest: Path, taken=()) -> str:
    """A clean, TITLE-based filename stem for a paper's artifacts.

    Titles read far better than DOI slugs in a folder listing.  Unique within
    the fulltext folder: a title collision with a different already-fetched
    paper (on disk) or an already-assigned one in ``taken`` (for a batch being
    fetched concurrently) falls back to a short key suffix.  Empty title -> the
    DOI/key slug.
    """
    title = (paper.get("title") or "").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:80].strip("-")
    if not slug:
        return _slug(key)
    collides = slug in taken or any(p.exists() for p in (
        dest / f"{slug}.pdf", dest / "xml" / f"{slug}.xml", dest / "txt" / f"{slug}.txt",
    ))
    return f"{slug}__{_slug(key)[:10]}" if collides else slug


def fetch_one(paper: dict, dest: Path, basename: str | None = None) -> dict | None:
    """Run the fallback chain for one paper; save verified bytes + text.

    ``basename`` may be pre-assigned by a concurrent caller (so parallel
    downloads do not race on title-collision detection); otherwise it is derived
    here.  Returns a manifest record on success, or None if no source yielded a
    verified full text.
    """
    arxiv = (paper.get("arxiv_id") or "").strip()
    doi = (paper.get("doi") or "").strip()
    pmcid = (paper.get("pmcid") or "").strip()
    oa_url = (paper.get("oa_pdf_url") or "").strip()

    attempts = []
    if arxiv:
        attempts.append(("arxiv", lambda: _fetch_arxiv(arxiv)))
    if doi.startswith("10.1101"):
        attempts.append(("biorxiv", lambda: _fetch_biorxiv(doi)))
    if pmcid:
        attempts.append(("europepmc", lambda: _fetch_europepmc(pmcid)))
    if oa_url:
        attempts.append(("oa_url", lambda: _fetch_direct_url(oa_url)))
    if doi:
        attempts.append(("unpaywall", lambda: _fetch_unpaywall(doi)))

    key = _fulltext_key(paper)
    if key is None:
        return None

    for source, fn in attempts:
        got = fn()
        if not got:
            continue
        content, fmt = got
        # Clean layout: PDFs at the top level; XML and extracted text in their
        # own subfolders; every file named by the paper TITLE.
        base = basename or _paper_basename(paper, key, dest)
        rel_file = f"xml/{base}.xml" if fmt == "xml" else f"{base}.{fmt}"
        src_path = dest / rel_file
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_bytes(content)
        text = _jats_to_text(content) if fmt == "xml" else _pdf_to_text(content)
        text_file = ""
        if text:
            text_file = f"txt/{base}.txt"
            text_path = dest / text_file
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text)
        logger.info("fulltext: fetched %s via %s (%s, %d bytes, %d chars)",
                    key, source, fmt, len(content), len(text))
        return {
            "key": key,
            "title": (paper.get("title") or "")[:200],
            "doi": doi, "pmcid": pmcid, "arxiv_id": arxiv,
            "source": source, "format": fmt,
            "file": rel_file, "bytes": len(content),
            "text_file": text_file, "text_chars": len(text),
            "verified": True,
        }
    return None


@tool(
    description=(
        "Download the ACTUAL full text (PDF or full-text XML) of the top-ranked "
        "discovered papers and save it under this run's artifacts/fulltext/ "
        "folder. Tries arXiv, bioRxiv, Europe PMC, the paper's open-access link, "
        "and Unpaywall, verifying each download is a real PDF/XML (not an HTML "
        "error page). Use AFTER consolidate_literature, to read decision-relevant "
        "evidence (kinetics, mechanism, engineering) that lives in full text, not "
        "abstracts. Bounded by max_papers; skips papers already fetched."
    ),
    category="reasoning",  # not 'literature': it reads, it is not a new search
    parallel_safe=False,
    injected_params=["run_dir", "llm_client"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "query": {
            "type": "string",
            "description": "Campaign intent used to rank which papers to fetch.",
        },
        "max_papers": {
            "type": "integer",
            "description": "Max papers to download this call (default 8).",
            "default": 8,
        },
    },
)
def fetch_full_text(query: str, run_dir: str | None = None,
                    max_papers: int = 8, llm_client=None) -> dict:
    """Fetch + store verified full text for the top-ranked discovered papers."""
    if not run_dir:
        return {"status": "error", "summary": "No run_dir available.", "data": []}

    # The model may hand max_papers back as a string ("8"); coerce so the
    # downstream slice / consolidate_pool call never sees a non-integer.
    try:
        max_papers = int(max_papers)
    except (TypeError, ValueError):
        max_papers = 8
    max_papers = max(1, max_papers)

    papers = _recover_literature_papers(run_dir)
    if not papers:
        return {
            "status": "partial",
            "summary": "No accumulated literature to fetch full text for.",
            "data": [],
            "count": 0,
            "suggestions": ["Run literature searches and consolidate first."],
        }

    ranked = consolidate_pool(query, papers, max_results=max_papers * 3)["ranked"]

    dest = Path(run_dir) / "artifacts" / "fulltext"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = []
    done = {m.get("key") for m in manifest}

    # Relevance gate BEFORE download: a semantic rank cannot tell an enzymatic
    # transformation from a purely chemical one on the same product, nor a
    # bond-forming reaction from its reverse — so it would pull off-topic and
    # opposite-direction papers whose deposited accessions then pollute the
    # candidate pool.  Gate each paper on title+abstract first; only download
    # what passes.  Fails OPEN (keeps the paper) when no LLM is available.
    from enzyme_discovery_agent.tools.literature_extraction import is_paper_relevant

    # De-duplicate candidates by full-text key and drop already-fetched papers so
    # concurrent workers never target the same output file or re-download.
    candidates, seen_keys = [], set()
    for paper in ranked:
        key = _fulltext_key(paper)
        if key is None or key in done or key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append((key, paper))

    def _gate_and_fetch(item):
        """Relevance-gate then download ONE paper (both per-paper network/LLM
        latency, independent across papers) — runs on a worker thread."""
        key, paper = item
        if llm_client is not None:
            ok, reason = is_paper_relevant(
                paper.get("title", ""), paper.get("abstract", ""), query, llm_client)
            if not ok:
                return "gated", key, (paper.get("title") or "")[:60], reason, None
        return "fetched", key, None, None, fetch_one(paper, dest)

    # Download + gate papers CONCURRENTLY instead of one at a time.  Work in waves
    # sized to the remaining need so we stop near max_papers without downloading
    # the whole (3x) candidate pool; a full successful wave is exactly `need`, so
    # the total never overshoots max_papers and no orphan files are left.
    # Aggregation stays on this thread, so manifest/done/fetched need no lock.
    fetched, examined, gated = [], 0, 0
    i = 0
    with ThreadPoolExecutor(max_workers=config.FULLTEXT_HARVEST_WORKERS) as pool:
        while i < len(candidates) and len(fetched) < max_papers:
            wave = candidates[i:i + max(1, max_papers - len(fetched))]
            i += len(wave)
            for kind, key, title, reason, rec in pool.map(_gate_and_fetch, wave):
                if kind == "gated":
                    gated += 1
                    logger.info("fetch_full_text relevance gate dropped %s: %s",
                                title, reason)
                    continue
                examined += 1
                if rec:
                    manifest.append(rec)
                    done.add(key)
                    fetched.append(rec)

    try:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except OSError as e:
        logger.warning("Failed to write fulltext manifest: %s", e)

    summary = (f"Downloaded full text for {len(fetched)} paper(s) "
               f"(examined {examined}, {gated} dropped as off-topic/"
               f"wrong-direction before download); {len(manifest)} total in this run.")
    return {
        "status": "success" if fetched else "partial",
        "summary": summary,
        "data": [
            {k: r[k] for k in ("title", "source", "format", "text_chars",
                               "file", "doi", "pmcid")}
            for r in fetched
        ],
        "count": len(fetched),
        "fulltext_dir": str(dest),
    }


def _resolve_embl_accession(acc: str, cap: int = 50) -> list[str]:
    """Deterministically map an EMBL/GenBank accession to UniProt accessions.

    Skips genome-scale deposits (hundreds of proteins) that would flood the
    registry, keeping cluster-scale deposits (tens) where the relevant enzyme
    lives.  A biosynthetic-gene-cluster accession resolves to the whole cluster
    (tens of proteins), so it must not be truncated to the resolver's default
    10, which could drop the target enzyme sitting deeper in the cluster.
    """
    params = {"query": f"xref:embl-{acc}", "format": "json",
              "size": str(cap + 1), "fields": "accession"}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    try:
        raw = request_with_retry(
            urllib.request.Request(url, headers={"Accept": "application/json"}),
            timeout=20,
        )
        results = json.loads(raw.decode()).get("results", [])
    except Exception as e:
        logger.debug("EMBL xref resolve failed for %s: %s", acc, e)
        return []
    if len(results) > cap:  # genome-scale deposit — too noisy to register
        return []
    return [r.get("primaryAccession") for r in results if r.get("primaryAccession")]


# A gene symbol: one letter then alphanumerics/-/_.  Multi-word protein names
# are resolved via protein_name, never gene:.
_GENE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,20}$")
# Bound on literature-NAMED enzymes resolved per run, and papers read for them.
_MAX_NAMED_ENZYMES = 40
_MAX_NAMED_PAPERS = 15


def _uniprot_search(query: str, size: int = 5, fields: str = "accession") -> list[dict]:
    params = {"query": query, "format": "json", "size": str(size), "fields": fields}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    try:
        raw = request_with_retry(
            urllib.request.Request(url, headers={"Accept": "application/json"}),
            timeout=20,
        )
        return json.loads(raw.decode()).get("results", [])
    except Exception as e:  # noqa: BLE001
        logger.debug("uniprot search failed (%s): %s", query[:60], e)
        return []


def _entry_gene_names(entry: dict) -> list[str]:
    out = []
    for g in entry.get("genes", []) or []:
        gn = (g.get("geneName") or {}).get("value")
        if gn:
            out.append(gn)
        for syn in g.get("synonyms", []) or []:
            if syn.get("value"):
                out.append(syn["value"])
    return out


def _entry_ecs(entry: dict) -> list[str]:
    rec = (entry.get("proteinDescription") or {}).get("recommendedName") or {}
    return [e.get("value") for e in rec.get("ecNumbers", []) or [] if e.get("value")]


# Generic words that carry no functional signal, so they must not make two
# unrelated proteins look coherent.
_NAME_STOPWORDS = frozenset({
    "protein", "putative", "probable", "uncharacterized", "hypothetical",
    "family", "domain", "containing", "type", "chain", "subunit", "like",
    "alpha", "beta", "gamma", "delta", "component", "related",
})


def _entry_name_tokens(entry: dict) -> set:
    rec = (entry.get("proteinDescription") or {}).get("recommendedName") or {}
    full = (rec.get("fullName") or {}).get("value", "")
    return {
        t for t in re.split(r"[^A-Za-z0-9]+", full.lower())
        if len(t) >= 4 and t not in _NAME_STOPWORDS
    }


def _functionally_coherent(entries: list[dict]) -> bool:
    """Do these entries denote ONE enzyme (the same function, which may span
    several organisms), rather than unrelated proteins sharing a gene symbol?

    A symbol can be reused for functionally different proteins across taxa, so a
    name is only trustworthy without an organism when the entries carrying it
    agree on function.  Coherent = they share an EC number, or a significant
    (non-generic) protein-name token; incoherent = a symbol collision, left for
    the organism-anchored path.
    """
    if len(entries) <= 1:
        return True
    ec_sets = [set(_entry_ecs(e)) for e in entries]
    if all(ec_sets) and set.intersection(*ec_sets):
        return True
    tok_sets = [t for t in (_entry_name_tokens(e) for e in entries) if t]
    return bool(tok_sets) and bool(set.intersection(*tok_sets))


def _resolve_named_enzyme(name: str, organism: str) -> str | None:
    """Resolve an enzyme NAMED in a paper's body to a UniProt accession, using
    the stated organism as a disambiguator.

    A paper commonly names target-reaction enzymes for which it deposits no
    sequence of its own (cited analogues / prior art).  The deposited-accession
    miner is blind to these (no accession in the text); this recovers them by
    name.

    A gene symbol can be reused for functionally unrelated proteins across taxa,
    so the policy is precision-first:
      * organism stated  -> organism-anchored query (precise; admits unreviewed);
      * organism absent   -> accept the reviewed (Swiss-Prot) entries carrying
        that symbol ONLY when they are functionally coherent (one enzyme, which
        may legitimately span several organisms); an incoherent symbol is left
        for the organism-anchored path.
    Returns a representative accession, or None when nothing resolves confidently.
    """
    name = (name or "").strip()
    organism = (organism or "").strip()
    if not name:
        return None
    is_gene = bool(_GENE_TOKEN_RE.match(name))
    esc = name.replace('"', " ")
    name_clause = f'(gene:{name} OR protein_name:"{esc}")' if is_gene else f'protein_name:"{esc}"'

    if organism:
        org = organism.replace('"', " ").strip()
        hits = _uniprot_search(f'{name_clause} AND organism_name:"{org}"', size=5)
        if hits:
            hits.sort(key=lambda e: 0 if "reviewed" in str(e.get("entryType", "")).lower() else 1)
            return hits[0].get("primaryAccession")

    if is_gene:
        reviewed = _uniprot_search(
            f"gene:{name} AND reviewed:true", size=10,
            fields="accession,gene_names,protein_name,ec",
        )
        exact = [
            r for r in reviewed
            if any(gn.lower() == name.lower() for gn in _entry_gene_names(r))
        ]
        if exact and _functionally_coherent(exact):
            return exact[0].get("primaryAccession")
    return None


def _register_named_enzymes_from_fulltext(
    run, query: str, dest: Path, manifest: list[dict], llm_client, register_fn,
    already: set, remaining: int,
) -> list[str]:
    """Literature-driven discovery channel: harvest enzymes NAMED in the fetched
    papers' body text and register each as a candidate.

    For each relevance-ranked paper, the LLM extracts the characterised enzymes
    and the cited/related enzymes (name + organism); each is resolved by
    name+organism (:func:`_resolve_named_enzyme`) and registered.  Literature
    builds on literature, so every fetched paper becomes a set of grounded leads
    — the route to an enzyme mentioned only by NAME (no accession in the text),
    which the deterministic accession miner cannot reach.  Needs an LLM; no-op
    without one.
    """
    if llm_client is None or remaining <= 0:
        return []
    from enzyme_discovery_agent.tools.literature_extraction import extract_from_paper
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry

    # Read each paper (one LLM extraction call) IN PARALLEL — the slow part —
    # then resolve + register SEQUENTIALLY so the remaining/dedup budget stays
    # correct.
    def _read_and_extract(meta):
        tf = meta.get("text_file")
        if not tf:
            return (meta, None)
        try:
            text = (dest / tf).read_text(errors="ignore")
        except OSError:
            return (meta, None)
        if not text.strip():
            return (meta, None)
        try:
            return (meta, extract_from_paper(text, "", "", llm_client, query=query))
        except Exception as e:  # noqa: BLE001 - never block mining
            logger.debug("named-enzyme extraction failed for %s: %s", tf, e)
            return (meta, None)

    with ThreadPoolExecutor(max_workers=config.FULLTEXT_HARVEST_WORKERS) as pool:
        extracted = list(pool.map(_read_and_extract, manifest[:_MAX_NAMED_PAPERS]))

    registered: list[str] = []
    paper_records: list[dict] = []
    for meta, obj in extracted:
        if len(registered) >= remaining:
            break
        if not obj:
            continue
        doi = meta.get("doi") or meta.get("key") or (meta.get("text_file") or "")
        enzymes = obj.get("enzymes") or []
        homologs = obj.get("homologs_or_related_enzymes") or []
        rec = {
            "title": meta.get("title", ""), "doi": doi,
            "source": meta.get("source", ""), "format": meta.get("format", ""),
            "enzymes": [], "homologs": [],
        }

        # (a) Enzymes whose sequence this paper DEPOSITED: attach the paper's
        #     name to the candidate the deposited-accession miner already
        #     registered (its UniProt annotation is often generic).  Enzymes
        # (b) named but not deposited here: resolve by name and register,
        #     carrying the paper's name as the candidate's label.
        # ``kind`` distinguishes an enzyme CHARACTERISED in this paper ("named":
        # the paper is directly relevant, a real evidence source) from one merely
        # CITED / named as a related analogue ("cited": a valid discovery LEAD,
        # but this paper does not characterise it — its evidence lives in its own
        # primary literature, reached later via its accession).
        def _handle(nm, org, gb, bucket, kind):
            nm = (nm or "").strip()
            org = org or ""
            accs = []
            if gb:
                accs = _resolve_embl_accession(gb)
                for acc in accs:
                    if nm:
                        update_candidate_registry(run, acc, None, {"literature_name": nm})
            elif nm:
                acc = _resolve_named_enzyme(nm, org)
                if acc:
                    accs = [acc]
                    if acc not in already and len(registered) < remaining:
                        already.add(acc)
                        register_fn(acc, f"fulltext_{kind}:{nm}@{doi}", nm)
                        registered.append(acc)
            bucket.append({"name": nm, "organism": org, "accessions": accs,
                           "relationship": kind})

        for e in enzymes:
            _handle(e.get("name"), e.get("organism"), (e.get("genbank") or "").strip(),
                    rec["enzymes"], "named")
        for h in homologs:
            _handle(h.get("name"), h.get("organism"), "", rec["homologs"], "cited")
        paper_records.append(rec)

    try:
        (dest / "papers_enzymes.json").write_text(json.dumps(paper_records, indent=2) + "\n")
    except OSError:
        pass
    return registered


def seed_mechanism_literature(
    run_dir, query: str, substrate_smiles: str = "", llm_client=None,
    max_families: int = 4, terms_per_family: int = 2, per_source: int = 15,
    log_fn=None,
) -> int:
    """Run the derived mechanism families' literature search phrases as REAL
    searches and persist the hits into the run's literature pool.

    Why: discovery otherwise depends on the model choosing mechanism-oriented
    search terms.  When it anchors on the query's applied PRODUCT instead, the
    literature pool fills with materials / application papers and the
    characterised enzymes — named only in mechanism primary literature — are never
    read, so the full-text harvest has nothing to mine.  Executing the families'
    own search phrases guarantees the mechanism literature is present before the
    harvest runs.  The families and their phrases are LLM-PROPOSED strategy (the
    tools merely execute them); no answer is encoded here.  Returns the number of
    papers added.  Never raises — a seeding failure must not block the harvest.
    """
    run = Path(run_dir)
    try:
        from enzyme_discovery_agent.tools.mechanism_families import (
            derive_mechanism_families,
        )
        mf_path = run / "mechanism_families.json"
        families: list[dict] = []
        if mf_path.exists():
            try:
                families = json.loads(mf_path.read_text())
            except (OSError, json.JSONDecodeError):
                families = []
        if not families and llm_client is not None:
            families = derive_mechanism_families(query, substrate_smiles, llm_client)
            if families:
                try:
                    mf_path.write_text(json.dumps(families, indent=2) + "\n")
                except OSError:
                    pass
    except Exception as exc:  # noqa: BLE001 - never block the harvest
        logger.warning("Mechanism-seed derivation failed: %s", exc)
        return 0
    if not families:
        return 0

    terms: list[str] = []
    seen_terms: set[str] = set()
    for fam in families[:max_families]:
        for t in (fam.get("literature_terms") or fam.get("descriptors") or [])[:terms_per_family]:
            t = (t or "").strip()
            if t and t.lower() not in seen_terms:
                seen_terms.add(t.lower())
                terms.append(t)
    if not terms:
        return 0

    from enzyme_discovery_agent.tools.literature import (
        search_europepmc,
        search_pubmed,
        search_semantic_scholar,
    )
    sources = (search_pubmed, search_europepmc, search_semantic_scholar)

    def _search_term(term: str) -> list[dict]:
        hits: list[dict] = []
        for fn in sources:
            try:
                r = fn(term, max_results=per_source)
                if r.get("status") != "error":
                    hits.extend(r.get("data") or [])
            except Exception as exc:  # noqa: BLE001 - one source failing is fine
                logger.debug("mechanism-seed search failed (%s): %s", term[:50], exc)
        return hits

    lit_dir = run / "artifacts" / "literature"
    lit_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    with ThreadPoolExecutor(max_workers=config.FULLTEXT_HARVEST_WORKERS) as pool:
        results = list(pool.map(_search_term, terms))
    for i, (term, hits) in enumerate(zip(terms, results)):
        papers = [p for p in hits if isinstance(p, dict)]
        if not papers:
            continue
        try:
            (lit_dir / f"mechanism_seed_{i:02d}.json").write_text(
                json.dumps({"tool": "mechanism_seed", "query": term,
                            "result": {"data": papers}}, indent=2) + "\n"
            )
            added += len(papers)
        except OSError:
            pass
    if log_fn and added:
        log_fn(f"  Mechanism-seed searches: added {added} paper(s) from "
               f"{len(terms)} mechanism search phrase(s) to the pool")
    return added


def mine_and_register_fulltext_accessions(
    run_dir, query: str, fetch_ceiling: int = 250, max_proteins: int = 400,
    log_fn=None, llm_client=None,
) -> list[str]:
    """Lean literature discovery: read a few open-access full texts and register the
    enzymes NAMED / CITED in them as discovered candidates.

    Two deliberate cost/quality choices (see the inline notes): (1) read only the
    top ``FULLTEXT_HARVEST_MAX_PAPERS`` DOWNLOADABLE papers in relevance order —
    paywalled papers fast-fail and are skipped — instead of LLM-relevance-gating and
    hammering the whole (mostly paywalled) pool; (2) do NOT regex-mine deposit-context
    accessions (that scooped wrong-direction / off-target enzymes for near-zero unique
    value).  Literature's grounded, irreplaceable channel is the LLM extraction of the
    enzymes a paper NAMES + CITES (``_register_named_enzymes_from_fulltext``), which
    reaches characterised enzymes invisible to database/family search.  No-op when
    ``llm_client`` is None (extraction needs it).
    """
    from enzyme_discovery_agent.models.candidate_registry import (
        update_candidate_registry,
    )

    run = Path(run_dir)
    papers = _recover_literature_papers(run)
    if not papers:
        return []

    dest = run / "artifacts" / "fulltext"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = []
    done = {m.get("key") for m in manifest}

    # 1. Read the full text of a SMALL number of the most-relevant DOWNLOADABLE
    # papers — the lean "exploration" harvest.  We deliberately DO NOT LLM-relevance-
    # gate the pool or attempt the whole (~10^2, mostly paywalled) list: that spent
    # dozens of gate calls + paywall round-trips to read a handful of open-access
    # papers.  Instead we walk the pool in relevance order and DOWNLOAD until
    # FULLTEXT_HARVEST_MAX_PAPERS full texts are in hand (non-OA papers fast-fail and
    # are skipped).  Reading an off-topic paper is cheap here: only the per-paper
    # named/cited extraction (step 2) registers anything, and it judges relevance
    # itself.  A bare-title non-article filter drops posters/proceedings up front.
    from enzyme_discovery_agent.tools.literature_extraction import (
        _is_non_article,
        select_papers_to_read,
    )
    from enzyme_discovery_agent.tools.paper_relevance import (
        assess_primary_evidence,
        rank_papers_by_enzymology,
    )
    workers = config.FULLTEXT_HARVEST_WORKERS
    target_reads = min(fetch_ceiling, config.FULLTEXT_HARVEST_MAX_PAPERS)
    # Let the AGENT REASON over titles+abstracts to pick which papers are most worth
    # reading IN FULL for the DERIVED target reaction: it recognises on-topic primary
    # biology and rejects chemical-synthesis / materials / review / off-topic papers
    # that merely share surface vocabulary — a judgement a keyword scorer cannot make.
    # Falls back to the cheap enzymology ranker (then the lexical+semantic consolidate
    # order) when no LLM is available or the call fails, so it never does worse.
    ranked = select_papers_to_read(
        run, papers, target_reads, llm_client, query,
        fallback=lambda: rank_papers_by_enzymology(
            run, papers,
            fallback=lambda: consolidate_pool(query, papers, max_results=len(papers))["ranked"],
        ),
    )
    candidates = [p for p in _select_harvest_papers(ranked, done, len(ranked))
                  if not _is_non_article(p.get("title", ""))]
    # Only when NO LLM did the reasoned selection above do we fall back to a cheap
    # SOFT filter: drop reviews / inhibitor-only papers while enough primary studies
    # remain to fill the budget (never starving a metadata-sparse pool).  With LLM
    # selection this is skipped — the model already excluded those, and its reasoned
    # picks must not be overridden by a keyword heuristic.
    if llm_client is None:
        primary = []
        for _p in candidates:
            _ev = assess_primary_evidence(_p)
            if not _ev["review"] and not _ev["inhibitor_only"]:
                primary.append(_p)
        if len(primary) >= target_reads:
            candidates = primary

    # Download in relevance order until `target_reads` open-access full texts are in
    # hand, in parallel batches (downloads are I/O-bound).  Bound the attempts so a
    # fully-paywalled pool cannot spin (open-access yield is typically ~1 in 4-8).
    taken: set = set()
    fetched = 0
    attempted = 0
    attempt_ceiling = min(len(candidates), max(target_reads * 8, 24))
    i = 0
    while fetched < target_reads and i < attempt_ceiling:
        batch = candidates[i:i + workers]
        if not batch:
            break
        i += len(batch)
        plan = []
        for p in batch:
            base = _paper_basename(p, _fulltext_key(p), dest, taken=taken)
            taken.add(base)
            plan.append((p, base))
        attempted += len(plan)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            recs = list(pool.map(lambda pb: fetch_one(pb[0], dest, basename=pb[1]), plan))
        for rec in recs:
            if rec and fetched < target_reads:
                manifest.append(rec)
                done.add(rec["key"])
                fetched += 1
    if log_fn:
        log_fn(f"  Full-text harvest: read {fetched} open-access paper(s) "
               f"(attempted {attempted} of {len(candidates)}) for named/cited-enzyme "
               f"discovery")
    if manifest:
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except OSError:
            pass

    # 2. Literature-NAMED / CITED enzymes (below) are literature's grounded discovery
    #    channel.  Deposit-context regex accession-mining is intentionally REMOVED: it
    #    scooped wrong-direction / off-target accessions (e.g. an inhibitor paper's
    #    target enzyme) into the candidate set for near-zero unique value.
    registered: list[str] = []

    def _register(acc: str, source: str, lit_name: str = ""):
        meta = {"source_tool": source}
        if lit_name:
            meta["literature_name"] = lit_name
        update_candidate_registry(run, acc, "discovered", metadata=meta)
        registered.append(acc)

    # 3. Literature-NAMED / CITED enzymes: a paper names + characterises target-
    #    reaction enzymes (and cites related analogues) it may deposit no sequence
    #    for.  Resolve each by name+organism and register — the only route to an
    #    enzyme mentioned by NAME with no accession in the text, and the path that
    #    reaches the answer's own biosynthetic neighbourhood.  LLM-gated over the few
    #    downloaded papers' bodies; skipped when no llm_client.
    n_named = 0
    if llm_client is not None and len(registered) < max_proteins:
        try:
            named = _register_named_enzymes_from_fulltext(
                run, query, dest, manifest, llm_client, _register,
                already=set(registered), remaining=max_proteins - len(registered),
            )
            n_named = len(named)
        except Exception as exc:  # never let discovery block mining
            logger.warning("named-enzyme discovery failed: %s", exc)

    if log_fn:
        log_fn(
            f"  Full-text mining: {fetched} open-access paper(s) read "
            f"-> {n_named} literature-named/cited enzyme(s) "
            f"-> {len(registered)} candidate(s) registered"
        )
    return registered


def write_fulltext_family_index(run_dir, log_fn=None) -> None:
    """Categorise the harvested full text BY ENZYME FAMILY.

    Joins the per-paper enzyme records (``papers_enzymes.json``, written during
    the harvest) to each resolved accession's family in the candidate registry,
    and writes two artifacts under ``artifacts/fulltext/``:
      * ``papers_enzymes.csv`` — one row per (paper, enzyme), sorted by family:
        family, enzyme, paper's name for it, organism, accession, role, title, doi;
      * ``families_index.json`` — family -> the papers/enzymes under it.
    Best run at finalization, when the registry carries each candidate's family.
    """
    import csv as _csv
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry

    run = Path(run_dir)
    dest = run / "artifacts" / "fulltext"
    try:
        papers = json.loads((dest / "papers_enzymes.json").read_text())
    except (OSError, json.JSONDecodeError):
        return
    reg = load_candidate_registry(run).get("candidates", {})

    def _meta(acc):
        return (reg.get(acc) or {}).get("metadata", {})

    rows = []
    for p in papers:
        for role in ("enzymes", "homologs"):
            for ez in p.get(role, []):
                accs = ez.get("accessions") or [""]
                for acc in accs:
                    meta = _meta(acc)
                    rows.append({
                        "family": meta.get("interpro_family") or "(unresolved)",
                        "enzyme": (reg.get(acc) or {}).get("enzyme_name") or ez.get("name", ""),
                        "paper_name_for_enzyme": ez.get("name", ""),
                        "organism": ez.get("organism", ""),
                        "accession": acc,
                        "role": role[:-1],
                        "paper_title": p.get("title", ""),
                        "doi": p.get("doi", ""),
                    })
    rows.sort(key=lambda r: (r["family"], r["paper_title"]))

    try:
        with open(dest / "papers_enzymes.csv", "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=[
                "family", "enzyme", "paper_name_for_enzyme", "organism",
                "accession", "role", "paper_title", "doi",
            ])
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        logger.warning("could not write papers_enzymes.csv: %s", exc)

    families: dict[str, list] = {}
    for r in rows:
        families.setdefault(r["family"], []).append({
            "enzyme": r["enzyme"], "accession": r["accession"],
            "paper_title": r["paper_title"], "doi": r["doi"],
        })
    try:
        (dest / "families_index.json").write_text(json.dumps(families, indent=2) + "\n")
    except OSError:
        pass
    if log_fn:
        log_fn(f"  Full-text family index: {len(rows)} enzyme mentions across "
               f"{len(families)} families -> papers_enzymes.csv")
