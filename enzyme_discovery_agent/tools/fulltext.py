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
    injected_params=["run_dir"],
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
                    max_papers: int = 8) -> dict:
    """Fetch + store verified full text for the top-ranked discovered papers."""
    if not run_dir:
        return {"status": "error", "summary": "No run_dir available.", "data": []}

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

    fetched, examined = [], 0
    for paper in ranked:
        if len(fetched) >= max_papers:
            break
        key = _fulltext_key(paper)
        if key is None or key in done:
            continue
        examined += 1
        rec = fetch_one(paper, dest)
        if rec:
            manifest.append(rec)
            done.add(key)
            fetched.append(rec)

    try:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except OSError as e:
        logger.warning("Failed to write fulltext manifest: %s", e)

    summary = (f"Downloaded full text for {len(fetched)} paper(s) "
               f"(examined {examined}); {len(manifest)} total in this run.")
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


# Sequence-accession patterns for full-text mining.  Deposited accessions live in
# the Methods / Data-Availability sections, never abstracts, and are the ONLY
# stable link to enzymes that carry no gene name and only a generic protein name
# (annotated merely as a generic ligase/AMP-binding activity) — which neither
# gene-name nor function-text search can reach.  Nucleotide accessions are
# 1-2 letters + 6 digits (older GenBank/ENA) or the newer 4-letters + 8/10 digits
# form; protein accessions are 3 letters + 5-7 digits.
_NUCLEOTIDE_ACC_RE = re.compile(r"\b([A-Z]{1,2}\d{6}|[A-Z]{4}\d{8,10})\b")
_PROTEIN_ACC_RE = re.compile(r"\b([A-Z]{3}\d{5,7})(?:\.\d+)?\b")
_DEPOSIT_CUES = (
    "accession", "deposited", "genbank", "ena ", "uniprot", "data availab",
    "nucleotide sequence", "embl", "ncbi", "gene cluster",
)


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

    registered: list[str] = []
    paper_records: list[dict] = []
    for meta in manifest[:_MAX_NAMED_PAPERS]:
        if len(registered) >= remaining:
            break
        tf = meta.get("text_file")
        if not tf:
            continue
        txt = dest / tf
        try:
            text = txt.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        doi = meta.get("doi") or meta.get("key") or txt.stem
        try:
            obj = extract_from_paper(text, "", "", llm_client, query=query)
        except Exception as e:  # noqa: BLE001 - never block mining
            logger.debug("named-enzyme extraction failed for %s: %s", tf, e)
            continue
        if not obj:
            continue
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
        def _handle(nm, org, gb, bucket):
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
                        register_fn(acc, f"fulltext_named:{nm}@{doi}", nm)
                        registered.append(acc)
            bucket.append({"name": nm, "organism": org, "accessions": accs})

        for e in enzymes:
            _handle(e.get("name"), e.get("organism"), (e.get("genbank") or "").strip(),
                    rec["enzymes"])
        for h in homologs:
            _handle(h.get("name"), h.get("organism"), "", rec["homologs"])
        paper_records.append(rec)

    try:
        (dest / "papers_enzymes.json").write_text(json.dumps(paper_records, indent=2) + "\n")
    except OSError:
        pass
    return registered


def mine_and_register_fulltext_accessions(
    run_dir, query: str, fetch_ceiling: int = 250, max_proteins: int = 400,
    log_fn=None, llm_client=None,
) -> list[str]:
    """Fetch mechanism-relevant papers' full text and register the sequence
    accessions deposited in them as discovered candidates.

    Accession mining itself is deterministic (regex + UniProt REST, no full text
    in the LLM context).  Which papers are mined is gated by a MECHANISM-relevance
    check on each paper's title+abstract (``is_paper_relevant``) — a lexical/
    semantic score cannot tell a synthesis paper from a degradation one, so it let
    off-topic papers through; the mechanism gate does not.  The gate needs an LLM;
    when ``llm_client`` is None it is skipped (all papers kept).  This is the only
    route to characterized-but-metadata-invisible enzymes tied to their paper
    solely through a deposited accession.
    """
    from enzyme_discovery_agent.models.candidate_registry import (
        UNIPROT_ACC_RE,
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

    # 1. Attempt full text for every MECHANISM-RELEVANT paper.  There is no count
    # cap (non-OA papers fast-fail); relevance is gated per paper by an LLM check
    # on the title+abstract, so an off-topic paper is not even downloaded and its
    # accessions never enter the pool.  fetch_ceiling is only a runaway guard.
    ranked = consolidate_pool(query, papers, max_results=len(papers))["ranked"]
    from enzyme_discovery_agent.tools.literature_extraction import is_paper_relevant
    fetched = 0
    dropped = 0
    for paper in ranked:
        if fetched >= fetch_ceiling:
            break
        key = _fulltext_key(paper)
        if key is None or key in done:
            continue
        if llm_client is not None:
            ok, reason = is_paper_relevant(
                paper.get("title", ""), paper.get("abstract", ""), query, llm_client)
            if not ok:
                dropped += 1
                logger.info("fulltext relevance gate dropped %s: %s",
                            (paper.get("title") or "")[:60], reason)
                continue
        rec = fetch_one(paper, dest)
        if rec:
            manifest.append(rec)
            done.add(key)
            fetched += 1
    if log_fn and dropped:
        log_fn(f"  Full-text relevance gate: dropped {dropped} off-topic paper(s) "
               f"before download")
    if manifest:
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except OSError:
            pass

    # 2. Mine deposit-context accessions from the stored full text (no LLM).
    #    Read via the manifest's relative text paths so it is agnostic to the
    #    on-disk layout (text lives under txt/ in the current layout).
    genbank: set[str] = set()
    uniprot: set[str] = set()
    cue_re = re.compile("|".join(re.escape(c) for c in _DEPOSIT_CUES))
    for m in manifest:
        tf = m.get("text_file")
        if not tf:
            continue
        try:
            text = (dest / tf).read_text(errors="ignore")
        except OSError:
            continue
        # Restrict matches to short windows around deposit cues: bounds the count
        # and eliminates the false positives a bare accession regex would hit.
        for m in cue_re.finditer(text.lower()):
            window = text[max(0, m.start() - 60): m.start() + 200]
            genbank.update(a.split(".")[0] for a in _NUCLEOTIDE_ACC_RE.findall(window))
            genbank.update(_PROTEIN_ACC_RE.findall(window))
            uniprot.update(UNIPROT_ACC_RE.findall(window))

    # 3. Register UniProt directly; resolve GenBank -> UniProt (cluster-scale).
    registered: list[str] = []

    def _register(acc: str, source: str, lit_name: str = ""):
        meta = {"source_tool": source}
        if lit_name:
            meta["literature_name"] = lit_name
        update_candidate_registry(run, acc, "discovered", metadata=meta)
        registered.append(acc)

    for acc in sorted(uniprot):
        if len(registered) >= max_proteins:
            break
        _register(acc, "fulltext_uniprot")
    for gb in sorted(genbank):
        if len(registered) >= max_proteins:
            break
        for acc in _resolve_embl_accession(gb):
            if len(registered) >= max_proteins:
                break
            _register(acc, f"fulltext_genbank:{gb}")

    # 4. Literature-NAMED enzymes: a paper names target-reaction enzymes it
    #    deposits no sequence for (cited analogues / prior art).  Resolve each by
    #    name+organism and register — the only route to an enzyme mentioned by
    #    NAME with no accession in the text.  LLM-gated; skipped when no
    #    llm_client.
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
            f"  Full-text mining: {fetched} papers fetched, "
            f"{len(genbank)} GenBank + {len(uniprot)} UniProt accessions "
            f"+ {n_named} literature-named enzyme(s) "
            f"-> {len(registered)} candidates registered"
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
