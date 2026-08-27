"""Extract structured EXPERIMENTAL MEASUREMENTS for enzymes from a paper's full text.

Motivation
----------
Quantitative activity data (kinetic parameters, specific activity, conversion/yield,
selectivity) is what lets a discovered enzyme be *ranked* and *validated*, but it is the
hardest thing to recover: the numbers usually live in TABLES and FIGURES, and the existing
full-text pipeline (``fulltext._jats_to_text``) flattens every table to a run of words,
destroying the row/column structure the numbers depend on. Worse, many journals ship the
key activity table as a raster IMAGE inside the article XML, so it is invisible to any
text parser.

This module adds three capabilities on top of the existing full-text plumbing:
  1. **Structured JATS table parsing** — turn ``<table-wrap>`` elements back into
     header + rows so a value keeps its (enzyme, substrate, metric) coordinates.
  2. **Image-only detection** — flag tables/figures that are graphics (no machine-readable
     cells) and expose their asset URLs, so the numbers are not silently dropped.
  3. **A measurement-focused extraction pass** (text tables via the LLM; image tables via
     an optional vision call) that returns one structured row per measurement with its
     provenance (which table/figure it came from).

Everything is reaction-agnostic: it extracts whatever quantitative enzyme data the paper
reports, with no assumption about the target chemistry.

Public entry point: the ``extract_experimental_measurements`` @tool. The parsing helpers
(``resolve_paper``, ``parse_jats_tables``, ``parse_jats_figures``) are import-safe and
network/LLM-free where noted, so they are unit-testable offline.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools.fulltext import (
    _get, _fetch_europepmc, _is_jats, _jats_to_text, _pdf_to_text, _fetch_unpaywall,
)

logger = logging.getLogger("enzyme_agent")

# Words that mark a table/figure as carrying quantitative enzyme measurements. Generic
# enzymology vocabulary only — no target-specific terms (see CLAUDE.md leakage firewall).
_MEASUREMENT_CUES = (
    "kcat", "k_cat", "turnover", "km", "k_m", "michaelis", "kcat/km", "catalytic efficiency",
    "specific activity", "conversion", "yield", "% conv", "relative activity", "residual activity",
    "kobs", "k_obs", "rate constant", "vmax", "v_max", "ki", "ic50", "ee", "enantiomeric",
    "selectivity", "activity", "kinetic", "u/mg", "µmol", "umol", "nmol", "min-1", "s-1",
)

# Metric vocabulary the extractor is allowed to emit. Generic across all enzymology.
_METRIC_ENUM = (
    "kcat", "Km", "kcat_over_Km", "specific_activity", "conversion_pct", "yield_pct",
    "relative_activity", "residual_activity", "kobs", "Vmax", "Ki", "IC50", "ee_pct",
    "selectivity", "other",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Table:
    label: str = ""                       # "Table 2"
    caption: str = ""
    header: list[list[str]] = field(default_factory=list)   # one list per header row
    rows: list[list[str]] = field(default_factory=list)     # body rows
    footnotes: str = ""
    is_image: bool = False                # True => cells live in a graphic, not markup
    graphic_href: str = ""                # asset id/href when is_image

    def to_markdown(self, max_rows: int = 60) -> str:
        out = []
        if self.label or self.caption:
            out.append(f"**{self.label}**: {self.caption}".strip(": ").strip())
        for hr in self.header:
            out.append(" | ".join(hr))
        if self.header:
            out.append(" | ".join("---" for _ in self.header[-1]))
        for r in self.rows[:max_rows]:
            out.append(" | ".join(r))
        if len(self.rows) > max_rows:
            out.append(f"... ({len(self.rows) - max_rows} more rows)")
        if self.footnotes:
            out.append(f"_footnotes_: {self.footnotes}")
        return "\n".join(out)

    def numeric_density(self) -> float:
        cells = [c for r in self.rows for c in r]
        if not cells:
            return 0.0
        numish = sum(1 for c in cells if re.search(r"\d", c))
        return numish / len(cells)

    def relevance(self) -> float:
        """Higher = more likely to hold enzyme measurements."""
        text = (self.label + " " + self.caption + " " + self.footnotes).lower()
        kw = sum(1 for c in _MEASUREMENT_CUES if c in text)
        return kw + 2.0 * self.numeric_density() + (0.0 if self.is_image else 0.0)


@dataclass
class Figure:
    label: str = ""
    caption: str = ""
    graphic_href: str = ""

    def relevance(self) -> float:
        text = (self.label + " " + self.caption).lower()
        return sum(1 for c in _MEASUREMENT_CUES if c in text)


# ---------------------------------------------------------------------------
# Namespace-agnostic XML helpers (JATS may or may not carry a default namespace)
# ---------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _findall_local(elem, name: str):
    return [e for e in elem.iter() if _local(e.tag) == name]


def _href_of(elem) -> str:
    """Return any xlink:href-style attribute value regardless of namespace prefix."""
    for k, v in elem.attrib.items():
        if _local(k) == "href":
            return v
    return ""


def _text_of(elem) -> str:
    return re.sub(r"\s+", " ", " ".join(elem.itertext())).strip()


def _int_attr(elem, name: str, default: int = 1) -> int:
    try:
        return max(1, int(elem.attrib.get(name, default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# JATS table + figure parsing (pure, offline, unit-testable)
# ---------------------------------------------------------------------------
def _parse_xhtml_table(table_elem) -> tuple[list[list[str]], list[list[str]]]:
    """Return (header_rows, body_rows) from a JATS/XHTML <table>. Best-effort colspan
    expansion; rowspan is not reconstructed (cell text is kept in its originating row)."""
    header, body = [], []
    theads = _findall_local(table_elem, "thead")
    tbodies = _findall_local(table_elem, "tbody")

    def row_cells(tr) -> list[str]:
        cells = []
        for c in tr:
            if _local(c.tag) not in ("td", "th"):
                continue
            span = _int_attr(c, "colspan", 1)
            txt = _text_of(c)
            cells.append(txt)
            for _ in range(span - 1):
                cells.append("")  # keep column alignment across the row
        return cells

    for th in theads:
        for tr in _findall_local(th, "tr"):
            header.append(row_cells(tr))
    body_src = tbodies if tbodies else [table_elem]
    seen_header = bool(header)
    for tb in body_src:
        for tr in _findall_local(tb, "tr"):
            cells = row_cells(tr)
            # If there was no <thead>, treat a leading all-<th> row as the header.
            if not seen_header and all(_local(c.tag) == "th" for c in tr if _local(c.tag) in ("td", "th")) and tr.findall("*"):
                header.append(cells)
                seen_header = True
            else:
                body.append(cells)
    return header, body


def parse_jats_tables(xml_bytes: bytes) -> list[Table]:
    """Extract every ``<table-wrap>`` as a structured :class:`Table`. Image-only tables
    (a ``<graphic>`` with no ``<table>``) are returned with ``is_image=True`` so the
    caller can route them to vision extraction rather than dropping them."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    tables: list[Table] = []
    for tw in _findall_local(root, "table-wrap"):
        label = next((_text_of(e) for e in tw if _local(e.tag) == "label"), "")
        caption = next((_text_of(e) for e in tw if _local(e.tag) == "caption"), "")
        foot = " ".join(_text_of(e) for e in _findall_local(tw, "table-wrap-foot"))
        inner = _findall_local(tw, "table")
        if inner:
            header, body = _parse_xhtml_table(inner[0])
            tables.append(Table(label=label, caption=caption, header=header,
                                rows=body, footnotes=foot))
        else:
            graphics = _findall_local(tw, "graphic") + _findall_local(tw, "inline-graphic")
            href = _href_of(graphics[0]) if graphics else ""
            tables.append(Table(label=label, caption=caption, footnotes=foot,
                                is_image=True, graphic_href=href))
    return tables


def parse_jats_figures(xml_bytes: bytes) -> list[Figure]:
    """Extract ``<fig>`` elements (label, caption, graphic href) — figures are the other
    common home of quantitative data (bar charts / heatmaps of activity or conversion)."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    figs: list[Figure] = []
    for fg in _findall_local(root, "fig"):
        label = next((_text_of(e) for e in fg if _local(e.tag) == "label"), "")
        caption = next((_text_of(e) for e in fg if _local(e.tag) == "caption"), "")
        graphics = _findall_local(fg, "graphic") + _findall_local(fg, "inline-graphic")
        href = _href_of(graphics[0]) if graphics else ""
        figs.append(Figure(label=label, caption=caption, graphic_href=href))
    return figs


# ---------------------------------------------------------------------------
# Paper resolution + full text retrieval (network)
# ---------------------------------------------------------------------------
_PMCID_RE = re.compile(r"^PMC\d+$", re.I)
_PMID_RE = re.compile(r"^\d{4,9}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def resolve_paper(identifier: str) -> dict:
    """Resolve a DOI / PMID / PMCID / free-text title to a record with a PMCID when one
    exists. Uses the Europe PMC core search. Returns {pmcid,pmid,doi,title,is_oa,source}."""
    ident = (identifier or "").strip()
    if _PMCID_RE.match(ident):
        return {"pmcid": ident.upper(), "pmid": "", "doi": "", "title": "", "is_oa": None,
                "source": "direct"}
    if _DOI_RE.match(ident):
        query = f'DOI:"{ident}"'
    elif _PMID_RE.match(ident):
        query = f"EXT_ID:{ident} AND SRC:MED"
    else:
        query = ident
    url = (f"{config.EUROPEPMC_API}/search?query={urllib.parse.quote(query)}"
           f"&resultType=core&format=json&pageSize=1")
    raw = _get(url, timeout=30)
    if not raw:
        return {"pmcid": "", "pmid": "", "doi": ident if _DOI_RE.match(ident) else "",
                "title": "", "is_oa": None, "source": "unresolved"}
    try:
        results = json.loads(raw).get("resultList", {}).get("result", [])
    except (json.JSONDecodeError, AttributeError):
        results = []
    if not results:
        return {"pmcid": "", "pmid": "", "doi": ident if _DOI_RE.match(ident) else "",
                "title": "", "is_oa": None, "source": "not_found"}
    r = results[0]
    return {
        "pmcid": (r.get("pmcid") or "").upper(),
        "pmid": r.get("pmid") or r.get("id") or "",
        "doi": r.get("doi") or "",
        "title": r.get("title") or "",
        "is_oa": (r.get("isOpenAccess") == "Y"),
        "source": "europepmc",
    }


def fetch_jats_or_text(paper: dict) -> tuple[bytes | None, str]:
    """Return (content_bytes, kind) where kind is 'xml' (JATS, tables intact), 'pdf-text'
    (flat text), or ''. Prefers Europe PMC JATS; falls back to Unpaywall OA PDF."""
    pmcid = paper.get("pmcid")
    if pmcid:
        got = _fetch_europepmc(pmcid)
        if got and got[1] == "xml" and _is_jats(got[0]):
            return got[0], "xml"
        if got and got[1] == "pdf":
            return got[0], "pdf"
    doi = paper.get("doi")
    if doi:
        got = _fetch_unpaywall(doi)
        if got:
            return got[0], got[1]
    return None, ""


def supp_zip_url(pmcid: str) -> str:
    """Europe PMC endpoint that returns a ZIP of every figure/table image + supplementary
    file for an open-access article — the reliable source for main-text table/figure images
    (per-image ``bin/`` URLs are inconsistent / bot-blocked across publishers)."""
    return f"{config.EUROPEPMC_API}/{pmcid}/supplementaryFiles"


# Per-paper ZIP cache so extracting several images from one article downloads it once.
_SUPP_ZIP_CACHE: dict[str, "bytes | None"] = {}


def _supp_zip_bytes(pmcid: str) -> bytes | None:
    if pmcid not in _SUPP_ZIP_CACHE:
        b = _get(supp_zip_url(pmcid), timeout=90)
        _SUPP_ZIP_CACHE[pmcid] = b if (b and b[:2] == b"PK") else None
    return _SUPP_ZIP_CACHE[pmcid]


def fetch_asset_bytes(pmcid: str, href: str) -> tuple[bytes | None, str]:
    """Return (image_bytes, filename) for a JATS graphic href by extracting the matching
    member from the article's supplementary ZIP. Prefers raster formats (jpg/png/gif) and
    matches on the href stem, tolerant of the extension the JATS href carries."""
    import io
    import zipfile
    if not pmcid or not href:
        return None, ""
    z = _supp_zip_bytes(pmcid)
    if not z:
        return None, ""
    try:
        zf = zipfile.ZipFile(io.BytesIO(z))
    except zipfile.BadZipFile:
        return None, ""
    stem = href.rsplit("/", 1)[-1]
    base = re.sub(r"\.(jpg|jpeg|png|gif|tif|tiff)$", "", stem, flags=re.I)
    names = zf.namelist()
    # Prefer an exact stem match, then same-base raster, preferring jpg > png > gif.
    def rank(n: str) -> int:
        nl = n.lower()
        ext = {"jpg": 0, "jpeg": 0, "png": 1, "gif": 2}.get(nl.rsplit(".", 1)[-1], 9)
        return ext
    cands = [n for n in names if re.sub(r"\.(jpg|jpeg|png|gif|tif|tiff)$", "", n.rsplit("/", 1)[-1], flags=re.I) == base
             and n.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg", "png", "gif")]
    if not cands:
        return None, ""
    pick = sorted(cands, key=rank)[0]
    try:
        return zf.read(pick), pick
    except KeyError:
        return None, ""


def image_asset_urls(pmcid: str, href: str) -> list[str]:
    """Provenance URLs for an image asset. The first is the ZIP the bytes are extracted
    from; the stem records which member. (Reporting only — fetching uses the ZIP.)"""
    if not pmcid or not href:
        return []
    return [supp_zip_url(pmcid), f"[member] {href.rsplit('/', 1)[-1]}"]


# ---------------------------------------------------------------------------
# LLM extraction (text tables) + optional vision (image tables)
# ---------------------------------------------------------------------------
_EXTRACT_SYS = (
    "You extract EXPERIMENTAL enzyme measurements from a paper's tables and text into "
    "structured JSON. Report only values actually stated (never infer or fill gaps). For "
    "each measurement emit one object with keys: enzyme (protein/variant name as written), "
    "variant (mutation string or 'WT' or ''), substrate (or ''), cofactor (or ''), "
    f"metric (one of {list(_METRIC_ENUM)}), value (number as written), unit (string), "
    "conditions (pH/temp/notes or ''), direction ('forward'/'reverse'/''), "
    "source (the table/figure label it came from). "
    'Return STRICT JSON: {"measurements":[ ... ]}. If none are present, return '
    '{"measurements":[]}. Do not wrap in prose or code fences.'
)


# A full activity matrix (enzymes × substrates × conditions) can be hundreds of rows;
# with this model's extended "thinking" tokens counted against the cap, the completion
# needs plenty of room or it stops (finish_reason=length) before emitting any answer.
# Output cap for one extraction call. A single table's measurements serialise to well under
# this; the old 20000 mostly paid for reasoning tokens on the vision path. 8000 comfortably
# fits ~100 measurement rows while cutting the per-call cost of the (parallel, vision-heavy)
# extraction jobs — the tool's dominant token sink.
_EXTRACT_MAX_TOKENS = 8000


def _llm_json(llm_client, system: str, user_parts: list, max_tokens: int = _EXTRACT_MAX_TOKENS) -> dict:
    """Call the injected OpenAI-compatible client and parse a JSON object from the reply.
    ``user_parts`` is an OpenAI 'content' list (text and/or image_url parts).

    Routes through ``LLMClient.call`` (retry + timeout) with a lifted ``max_tokens`` and
    ``temperature=None`` — this model rejects ``temperature`` as deprecated (400) and its
    reasoning tokens otherwise exhaust a small default cap.  The system instruction is
    folded into the user turn: a separate ``system`` role alongside an image reliably
    returned empty completions on the Argo/Vertex gateway."""
    if llm_client is None:
        from enzyme_discovery_agent.llm_client import LLMClient
        llm_client = LLMClient()
    parts = list(user_parts)
    if parts and parts[0].get("type") == "text":
        parts = [{"type": "text", "text": system + "\n\n" + parts[0]["text"]}] + parts[1:]
    else:
        parts = [{"type": "text", "text": system}] + parts
    resp = llm_client.call([{"role": "user", "content": parts}],
                           temperature=None, max_tokens=max_tokens)
    txt = (resp.choices[0].message.content if resp else "") or ""
    m = re.search(r"\{.*\}", txt, re.S)  # tolerant of ```json fences around the object
    if not m:
        return {"measurements": []}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"measurements": []}


def extract_measurements_from_tables(tables: list[Table], context: str,
                                     llm_client=None, max_tables: int = 6) -> list[dict]:
    """LLM-extract measurements from the structured (text) tables, richest first."""
    text_tables = sorted([t for t in tables if not t.is_image],
                         key=lambda t: t.relevance(), reverse=True)[:max_tables]
    if not text_tables:
        return []
    blob = "\n\n".join(t.to_markdown() for t in text_tables)
    user = [{"type": "text",
             "text": f"Paper context: {context[:1500]}\n\nTABLES:\n{blob}"}]
    out = _llm_json(llm_client, _EXTRACT_SYS, user)
    return out.get("measurements", []) if isinstance(out, dict) else []


def extract_measurements_from_image(pmcid: str, table: Table, context: str,
                                    llm_client=None) -> tuple[list[dict], str]:
    """Vision fallback for an image-only table/figure. Returns (measurements, status).
    Degrades gracefully to ('', 'vision_unavailable') if the endpoint rejects images."""
    img, member = fetch_asset_bytes(pmcid, table.graphic_href)
    if not img:
        return [], "image_not_fetched"
    mime = ("image/png" if member.lower().endswith("png")
            else "image/gif" if member.lower().endswith("gif") else "image/jpeg")
    data_uri = f"data:{mime};base64," + base64.b64encode(img).decode()
    user = [
        {"type": "text", "text": (f"Paper context: {context[:800]}\n"
                                  f"This image is {table.label}: {table.caption}\n"
                                  "Read every numeric enzyme measurement in it.")},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    try:
        out = _llm_json(llm_client, _EXTRACT_SYS, user)
    except Exception as e:  # noqa: BLE001 - image support is provider-dependent
        logger.debug("vision extraction failed: %s", e)
        return [], "vision_unavailable"
    return (out.get("measurements", []) if isinstance(out, dict) else []), "ok"


# ---------------------------------------------------------------------------
# Parallel fan-out: the per-image/-figure/-table LLM calls are independent and
# I/O-bound (remote inference), so running them concurrently collapses N slow
# sequential calls into ~one call's wall-time. Bounded workers keep within the
# Argo per-host rate limit (the HTTP layer throttles on top of this).
# ---------------------------------------------------------------------------
_PARALLEL_WORKERS = 6


def _run_jobs(jobs: list, max_workers: int = _PARALLEL_WORKERS) -> list:
    """Run zero-arg callables concurrently; results returned in submission order.
    A job that raises is captured as ('__error__', exc) rather than failing the batch."""
    if not jobs:
        return []
    results: list = [None] * len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as ex:
        futs = {ex.submit(fn): i for i, fn in enumerate(jobs)}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 - isolate one job's failure
                results[i] = ("__error__", e)
    return results


# ---------------------------------------------------------------------------
# Flat-PDF (no JATS) path: a PDF's tables are laid out graphically, so flattening to
# text scrambles their columns and a long table can fall outside any single text window.
# Instead, render the pages richest in measurement cues to images and read them via vision
# — the same capability used for JATS image tables, applied page-wise to the PDF.
# ---------------------------------------------------------------------------
def _pdf_cue_page_images(pdf_bytes: bytes, max_pages: int = 6,
                         zoom: float = 3.0) -> list[tuple[int, str, float]]:
    """Render the measurement-cue-richest PDF pages to PNG data-URIs for vision extraction.
    Returns [(page_index, data_uri, cue_score)], richest first. Empty if PyMuPDF is
    unavailable or the bytes are not a parseable PDF (caller then falls back to flat text)."""
    try:
        import pymupdf  # PyMuPDF; lazy so the module imports without it
    except Exception:  # noqa: BLE001 - optional dependency
        return []
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:  # noqa: BLE001 - corrupt/non-PDF bytes
        return []
    scored: list[tuple[int, float]] = []
    for i in range(doc.page_count):
        t = doc[i].get_text().lower()
        cue = sum(t.count(c) for c in _MEASUREMENT_CUES)
        if "table" in t:
            cue += 3  # a page that labels a table is a strong candidate
        if cue > 0:
            scored.append((i, float(cue)))
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[tuple[int, str, float]] = []
    for i, score in scored[:max_pages]:
        pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        uri = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
        out.append((i, uri, score))
    doc.close()
    return out


def extract_measurements_from_pdf_pages(page_images: list[tuple[int, str, float]],
                                        context: str, llm_client=None) -> tuple[list[dict], list[dict]]:
    """Vision-extract measurements from rendered PDF page images (one parallel job per page).
    Returns (measurements, per_page_notes)."""
    def _job(idx: int, uri: str):
        def run():
            user = [
                {"type": "text",
                 "text": (f"Paper context: {context[:800]}\n"
                          f"This is page {idx + 1} of a PDF article. Read EVERY numeric enzyme "
                          "measurement in any table on this page (kinetics, activity, "
                          "conversion, selectivity), keeping each value with its enzyme and "
                          "substrate.")},
                {"type": "image_url", "image_url": {"url": uri}},
            ]
            try:
                out = _llm_json(llm_client, _EXTRACT_SYS, user)
            except Exception as e:  # noqa: BLE001 - image support is provider-dependent
                logger.debug("pdf-page vision failed p%d: %s", idx, e)
                return ([], "vision_unavailable", idx)
            rows = out.get("measurements", []) if isinstance(out, dict) else []
            return (rows, "ok", idx)
        return run

    results = _run_jobs([_job(i, uri) for (i, uri, _s) in page_images])
    measurements, notes = [], []
    for res in results:
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "__error__":
            notes.append({"page": None, "status": "error", "n": 0})
            continue
        rows, status, idx = res
        notes.append({"page": idx + 1, "status": status, "n": len(rows)})
        for m in rows:
            m.setdefault("provenance", {})["pdf_page"] = idx + 1
            measurements.append(m)
    return measurements, notes


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------
@tool(
    description=(
        "Extract structured EXPERIMENTAL measurements for enzymes (kinetic parameters, "
        "specific activity, conversion/yield, selectivity) from one paper's full text. "
        "Resolves a DOI / PMID / PMCID / title to open-access full text, parses the "
        "article's tables into rows (preserving enzyme/substrate/metric structure), and "
        "flags tables or figures that are images so their numbers are not lost. Returns "
        "one row per measurement with its source table/figure. Use when a candidate needs "
        "quantitative activity evidence to rank or validate it."
    ),
    category="literature",
    parallel_safe=True,
    injected_params=["llm_client"],
    params={
        "identifier": {
            "type": "string",
            "description": "DOI, PMID, PMCID, or paper title to pull measurements from.",
        },
        "context_hint": {
            "type": "string",
            "description": "Optional short note on which enzyme/measurement to focus on.",
            "default": "",
        },
        "try_images": {
            "type": "boolean",
            "description": "Also attempt to read image-only tables/figures via vision (default true).",
            "default": True,
        },
    },
)
def extract_experimental_measurements(identifier: str, context_hint: str = "",
                                      try_images: bool = True, llm_client=None) -> dict:
    """Resolve a paper, parse its tables, and extract structured enzyme measurements."""
    paper = resolve_paper(identifier)
    content, kind = fetch_jats_or_text(paper)
    if not content:
        return {"status": "partial",
                "summary": f"No open-access full text found for '{identifier}'.",
                "data": [], "count": 0, "paper": paper,
                "suggestions": ["Try a PMCID directly, or a different open-access mirror."]}

    if kind != "xml":
        # No JATS. Render the measurement-bearing PDF pages and read them via vision — a PDF
        # table is a graphic, so flat text scrambles its columns and a long table can fall
        # outside any single text window. Fall back to a LARGE flat-text pass if PyMuPDF is
        # unavailable or vision finds nothing.
        text = _pdf_to_text(content) if kind == "pdf" else ""
        context = context_hint or paper.get("title", "") or (text[:1200] if text else "")
        page_imgs = _pdf_cue_page_images(content) if (kind == "pdf" and try_images) else []
        measurements, vision_notes = [], []
        if page_imgs:
            measurements, vision_notes = extract_measurements_from_pdf_pages(
                page_imgs, context, llm_client)
        if not measurements and text:
            # No pages rendered (no PyMuPDF / no cue pages) or vision empty: flat-text fallback
            # over a wide window (not the old 4k cap, which missed tables deep in the paper).
            measurements = extract_measurements_from_tables(
                [Table(label="(full text)", caption=text[:16000])], context, llm_client)
        return {"status": "ok" if measurements else "partial",
                "summary": (f"No JATS (kind={kind}); extracted {len(measurements)} "
                            f"measurement(s) from {len(page_imgs)} rendered page(s) via vision"
                            + ("" if page_imgs else " / flat text") + "."),
                "data": measurements, "count": len(measurements), "paper": paper,
                "image_tables": [],
                "pdf_pages_read": [i + 1 for (i, _u, _s) in page_imgs],
                "vision": vision_notes}

    tables = parse_jats_tables(content)
    figures = parse_jats_figures(content)
    context = context_hint or paper.get("title", "") or _jats_to_text(content)[:1200]
    pmcid = paper.get("pmcid")

    image_tables = [t for t in tables if t.is_image]
    # A single strong cue (e.g. "percent conversion", "kcat", "specific activity") in a
    # figure caption is enough to surface it as a data figure worth vision extraction.
    relevant_figs = [f for f in figures if f.relevance() >= 1]

    # Build independent extraction jobs and run them CONCURRENTLY: one job over all the
    # machine-readable text tables, plus one vision job per image-only table and per data
    # figure (top-ranked, capped). Each job is tagged so provenance can be attached on merge.
    jobs, tags = [], []
    if any(not t.is_image for t in tables):
        jobs.append(lambda: extract_measurements_from_tables(tables, context, llm_client))
        tags.append(("text", "jats_table"))
    if try_images and pmcid:
        for t in sorted(image_tables, key=lambda t: t.relevance(), reverse=True)[:3]:
            jobs.append(lambda a=t: extract_measurements_from_image(pmcid, a, context, llm_client))
            tags.append(("image", t.label))
        for f in sorted(relevant_figs, key=lambda f: f.relevance(), reverse=True)[:3]:
            jobs.append(lambda a=f: extract_measurements_from_image(pmcid, a, context, llm_client))
            tags.append(("figure", f.label))

    results = _run_jobs(jobs)

    measurements, vision_notes = [], []
    for (kind, label), res in zip(tags, results):
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "__error__":
            vision_notes.append({"asset": label, "kind": kind, "status": "error", "n": 0})
            continue
        if kind == "text":
            rows, extractor = (res or []), "jats_table"
        else:
            rows, status = res if isinstance(res, tuple) else (res, "ok")
            extractor = "vision"
            vision_notes.append({"asset": label, "kind": kind, "status": status, "n": len(rows)})
        for m in rows:
            m.setdefault("provenance", {})
            m["provenance"].update({"pmcid": pmcid, "doi": paper.get("doi"),
                                    "extractor": extractor, "source": label})
            measurements.append(m)

    summary = (f"{len(measurements)} measurement(s) from {pmcid or identifier} "
               f"[{len(jobs)} jobs in parallel]: "
               f"{len([t for t in tables if not t.is_image])} text table(s), "
               f"{len(image_tables)} image table(s), {len(relevant_figs)} data figure(s).")
    return {
        "status": "ok" if measurements else "partial",
        "summary": summary,
        "data": measurements,
        "count": len(measurements),
        "paper": paper,
        "image_tables": [{"label": t.label, "caption": t.caption,
                          "asset_urls": image_asset_urls(paper.get("pmcid", ""), t.graphic_href)}
                         for t in image_tables],
        "data_figures": [{"label": f.label, "caption": f.caption} for f in relevant_figs],
        "vision": vision_notes,
    }
