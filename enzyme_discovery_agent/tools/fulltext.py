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


def fetch_one(paper: dict, dest: Path) -> dict | None:
    """Run the fallback chain for one paper; save verified bytes + text.

    Returns a manifest record on success, or None if no source yielded a
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
        fname = f"{_slug(key)}.{fmt}"
        (dest / fname).write_bytes(content)
        text = _jats_to_text(content) if fmt == "xml" else _pdf_to_text(content)
        text_file = ""
        if text:
            text_file = f"{_slug(key)}.txt"
            (dest / text_file).write_text(text)
        logger.info("fulltext: fetched %s via %s (%s, %d bytes, %d chars)",
                    key, source, fmt, len(content), len(text))
        return {
            "key": key,
            "title": (paper.get("title") or "")[:200],
            "doi": doi, "pmcid": pmcid, "arxiv_id": arxiv,
            "source": source, "format": fmt,
            "file": fname, "bytes": len(content),
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
# (e.g. SfaB / A0A2H4T912, annotated merely "AMP-ligase") — which neither
# gene-name nor function-text search can reach.  Nucleotide accessions are
# 1-2 letters + 6 digits (KY427327) or the newer 4+8/10 form; protein accessions
# are 3 letters + 5-7 digits.
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
    (e.g. KY427327 -> 18 proteins incl. SfaB), so it must not be truncated to the
    resolver's default 10, which would drop the target.
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


def mine_and_register_fulltext_accessions(
    run_dir, query: str, fetch_ceiling: int = 250, max_proteins: int = 400, log_fn=None,
) -> list[str]:
    """Fetch top mechanism-relevant papers' full text and register the sequence
    accessions deposited in them as discovered candidates.

    Fully deterministic — regex over downloaded text plus UniProt REST resolution,
    with NO full text ever placed in the LLM context, so it adds no prompt tokens.
    This is the only route to characterized-but-metadata-invisible enzymes such as
    SfaB, whose identity is tied to its paper solely through a deposited GenBank
    accession.  Resolved candidates flow into the normal screen/evaluate pipeline,
    where the non-enzyme filter and mechanism ranking handle the cluster context.
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

    # 1. Attempt full text for every RELEVANT paper.  There is no count cap
    # (mining is token-free and non-OA papers fast-fail), but relevance IS gated:
    # only papers clearing FULLTEXT_MINE_MIN_SCORE are mined, so off-topic papers
    # never contribute accessions.  fetch_ceiling is only a runaway guard.
    ranked = consolidate_pool(query, papers, max_results=len(papers))["ranked"]
    relevant = [
        p for p in ranked
        if p.get("semantic_score") is None
        or p.get("semantic_score", 0.0) >= config.FULLTEXT_MINE_MIN_SCORE
    ]
    fetched = 0
    for paper in relevant:
        if fetched >= fetch_ceiling:
            break
        key = _fulltext_key(paper)
        if key is None or key in done:
            continue
        rec = fetch_one(paper, dest)
        if rec:
            manifest.append(rec)
            done.add(key)
            fetched += 1
    if manifest:
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except OSError:
            pass

    # 2. Mine deposit-context accessions from the stored full text (no LLM).
    genbank: set[str] = set()
    uniprot: set[str] = set()
    cue_re = re.compile("|".join(re.escape(c) for c in _DEPOSIT_CUES))
    for txt in dest.glob("*.txt"):
        try:
            text = txt.read_text(errors="ignore")
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

    def _register(acc: str, source: str):
        update_candidate_registry(run, acc, "discovered", metadata={"source_tool": source})
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

    if log_fn:
        log_fn(
            f"  Full-text mining: {fetched} papers fetched, "
            f"{len(genbank)} GenBank + {len(uniprot)} UniProt accessions "
            f"-> {len(registered)} candidates registered"
        )
    return registered
