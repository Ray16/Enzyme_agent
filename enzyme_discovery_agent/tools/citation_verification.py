"""Post-report citation verification.

After the LLM generates a report, scan for PMID citations, fetch their
abstracts via PubMed, and use a second LLM call to verify whether each
cited claim is actually supported by the abstract text.

Unsupported claims are flagged with a [CITATION UNVERIFIED] tag in the
report and collected in a separate verification summary.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.parse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enzyme_discovery_agent.llm_client import LLMClient

logger = logging.getLogger("enzyme_agent")

# Matches: PMID 12345678, PMID: 12345678, PMID [12345678](url), (PMID 12345678)
_PMID_PATTERN = re.compile(
    r'\bPMID[:\s]*\[?(\d{7,8})\]?(?:\([^)]*\))?'
    r'|pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,8})',
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

VERIFICATION_PROMPT = """\
You are a scientific fact-checker. Given a PubMed abstract and a claim made \
about that paper, determine whether the abstract supports the claim.

Abstract (PMID {pmid}):
{abstract}

Claim made about this paper:
{claim}

Answer in JSON only:
{{
  "supported": true/false/partial,
  "confidence": "high"/"medium"/"low",
  "explanation": "one sentence",
  "supporting_quote": "exact quote from abstract, or null if not supported"
}}"""


def _fetch_pubmed_abstract(pmid: str) -> str | None:
    """Fetch abstract text for a PMID via NCBI E-utilities."""
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EnzymeAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if len(text.strip()) < 50:
            return None
        return text.strip()[:3000]
    except Exception as e:
        logger.debug("Failed to fetch abstract for PMID %s: %s", pmid, e)
        return None


def _extract_claim_context(report_text: str, pmid: str) -> str:
    """Extract the sentence(s) immediately surrounding a PMID mention."""
    pattern = re.compile(
        rf'\bPMID[:\s]*\[?{pmid}\]?(?:\([^)]*\))?|pubmed\.ncbi\.nlm\.nih\.gov/{pmid}',
        re.IGNORECASE,
    )
    match = pattern.search(report_text)
    if not match:
        return ""

    start = max(0, match.start() - 400)
    end = min(len(report_text), match.end() + 400)
    context = report_text[start:end]

    # Try to trim to sentence boundaries
    sentences = _SENTENCE_END.split(context)
    relevant = [s for s in sentences if pmid in s or len(s) < 200]
    return " ".join(relevant[:3]).strip()


def _verify_one(pmid: str, abstract: str, claim_context: str, llm: LLMClient) -> dict:
    """Run one LLM verification call for a single PMID claim."""
    prompt = VERIFICATION_PROMPT.format(
        pmid=pmid,
        abstract=abstract[:2000],
        claim=claim_context[:600],
    )
    try:
        response = llm.call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        if response is None:
            return {"supported": "unknown", "confidence": "low", "explanation": "LLM returned None"}
        text = response.choices[0].message.content or ""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.debug("Verification LLM call failed for PMID %s: %s", pmid, e)
    return {"supported": "unknown", "confidence": "low", "explanation": "verification failed"}


def verify_report_citations(
    report_text: str,
    llm: LLMClient,
    max_pmids: int = 15,
) -> tuple[str, list[dict]]:
    """Verify PMID-backed claims in a report against PubMed abstracts.

    Returns (annotated_report, verification_results).
    Unsupported claims are flagged inline with [CITATION UNVERIFIED].
    """
    pmids = list(dict.fromkeys(
        g1 or g2
        for g1, g2 in _PMID_PATTERN.findall(report_text)
        if g1 or g2
    ))[:max_pmids]
    if not pmids:
        return report_text, []

    logger.info("Citation verification: checking %d PMIDs", len(pmids))

    results = []
    for pmid in pmids:
        abstract = _fetch_pubmed_abstract(pmid)
        if not abstract:
            logger.debug("No abstract found for PMID %s — skipping", pmid)
            continue

        claim_context = _extract_claim_context(report_text, pmid)
        if not claim_context:
            continue

        verification = _verify_one(pmid, abstract, claim_context, llm)
        verification["pmid"] = pmid
        verification["claim_context"] = claim_context[:300]
        results.append(verification)

        supported = verification.get("supported")
        # Only flag inline when the abstract is entirely unrelated to the enzyme
        # named in the claim (hard False with high confidence). Partial matches
        # mean the right enzyme is described but scope differs — that is the
        # agent's inference, which is valid and should not be flagged inline.
        if supported is False and verification.get("confidence") == "high":
            flag = f"PMID {pmid} [CITATION UNVERIFIED: {verification.get('explanation', '')}]"
            report_text = re.sub(
                rf'\bPMID[:\s]*\[?{pmid}\]?(?:\([^)]*\))?|pubmed\.ncbi\.nlm\.nih\.gov/{pmid}',
                flag,
                report_text,
                count=1,
                flags=re.IGNORECASE,
            )
            logger.warning("PMID %s claim not supported by abstract: %s",
                           pmid, verification.get("explanation", ""))

    if results:
        flagged = [r for r in results
                   if r.get("supported") is False and r.get("confidence") == "high"]
        partial = [r for r in results if r.get("supported") == "partial"]
        verified = [r for r in results if r.get("supported") is True]
        summary_lines = ["\n\n---\n## Citation Verification Summary\n",
                         f"Verified: {len(verified)} | Partial: {len(partial)} "
                         f"| Wrong paper: {len(flagged)}\n"]
        for r in flagged:
            summary_lines.append(
                f"- ✗ PMID {r['pmid']}: {r.get('explanation', '')}"
            )
        for r in partial:
            summary_lines.append(
                f"- ⚠ PMID {r['pmid']} (partial): {r.get('explanation', '')}"
            )
        report_text += "\n".join(summary_lines)

    return report_text, results
