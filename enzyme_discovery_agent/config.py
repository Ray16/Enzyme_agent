"""Configuration constants for the enzyme discovery agent."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader.

    Populates ``os.environ`` from a gitignored ``.env`` at the repository root
    WITHOUT overriding variables already present in the real environment (so an
    exported var always wins).  This keeps secrets such as API keys out of
    source control while still making them available to the agent.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        if not env_path.is_file():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

# --- LLM ---
ARGO_BASE_URL = os.environ.get("ARGO_BASE_URL", "https://apps-dev.inside.anl.gov/argoapi/v1")
ARGO_API_KEY = os.environ.get("ARGO_API_KEY", os.environ.get("USER", "rzhu"))
DEFAULT_MODEL = "argo:claude-5-sonnet"
LLM_TIMEOUT = 120  # seconds per LLM call
# Sonnet 5 rejects the legacy OpenAI ``temperature`` parameter.  Leaving it
# unspecified uses the provider default and remains compatible with Argo.
LLM_TEMPERATURE = None

# --- Agent defaults ---
MAX_ITERATIONS = 100
REFLECTION_INTERVAL = 5  # reflect every N iterations
# Reserve the final portion of every campaign for deterministic evaluation and
# LLM report generation.  Discovery is valuable only if it reaches deliverables.
FINALIZATION_RESERVE_ITERATIONS = 8

# --- Tool execution ---
TOOL_THREAD_POOL_WORKERS = 4
MAX_RESULT_CHARS = 20_000
# Breadth is gathered as literature/database leads; EC annotations remain
# compatibility gates, never a source of bulk candidates.  A larger but still
# auditable, evidence-led shortlist is then evaluated across distinct folds.
# (The EC-to-UniProt bulk retrieval path was removed rather than zeroed out;
# see gates/systematic_coverage.py.)
MAX_SCREEN_REPRESENTATIVES = 20
MAX_EXPLORATORY_FAMILY_REPRESENTATIVES = 5
EVIDENCE_SELECTION_SEARCH_THRESHOLD = 16
MIN_LITERATURE_SEARCH_ROUNDS = 3

# --- LLM context compaction ---
ABSTRACT_TRUNCATE_CHARS = 200
DESCRIPTION_TRUNCATE_CHARS = 200
PAPER_LIST_CAP = 8
RESULTS_LIST_CAP = 10

# --- Literature search ---
OVERFETCH_RATIO = 3
RERANK_MIN_SCORE = 0.25
# Keyword-result order is the reliable baseline.  Semantic re-ranking is
# opt-in because its Torch/Transformers import can stall or destabilize an
# otherwise network-bound discovery run.
ENABLE_SEMANTIC_RERANK = os.environ.get("ENZYME_AGENT_ENABLE_SEMANTIC_RERANK", "0") == "1"

# --- Database search ---
KEGG_EC_FETCH_LIMIT = 25
KEGG_THREAD_POOL_WORKERS = 5
INTERPRO_PAGE_SIZE_CAP = 50
UNIPROT_MAX_RESULTS_CAP = 500

# --- Reasoning / evaluation ---
SCREENING_BATCH_SIZE = 10
ENGINEERING_PAPER_WEIGHT = 2
WELL_CHARACTERIZED_THRESHOLD = 5
ENGINEERING_PRECEDENT_THRESHOLD = 2
EC_MATCH_DEPTH_STRONG = 3
EC_MATCH_DEPTH_PARTIAL = 3
LOGP_PENALTY_DIVISOR = 5.0
GEOMETRIC_MEAN_FLOOR = 1e-6

# Substrate size mismatch detection
MW_SEVERE_MISMATCH_THRESHOLD = 0.15   # ~6.7x MW ratio
MW_MODERATE_MISMATCH_THRESHOLD = 0.30  # ~3.3x MW ratio

# --- Report generation ---
# Keep Sonnet 5 compatible: it rejects the legacy OpenAI temperature parameter.
EXPLORATIVE_REPORT_TEMPERATURE = None

# --- Contact identity for API "polite pools" ---
# A real contact email is REQUIRED by Unpaywall (it 422-blocks example.com
# addresses) and doubles as the polite-pool ``mailto`` for OpenAlex, Crossref,
# and Europe PMC.  Overridable via env for other deployments.
CONTACT_EMAIL = os.environ.get("ENZYME_AGENT_CONTACT_EMAIL", "palm.uchicago@gmail.com")

# A browser-like User-Agent.  bioRxiv (Cloudflare) and some publishers reject
# default library UAs; sending one avoids 403/challenge responses.
HTTP_USER_AGENT = os.environ.get(
    "ENZYME_AGENT_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 EnzymeDiscoveryAgent/0.1",
)

# --- API endpoints ---
UNIPROT_API = "https://rest.uniprot.org"
KEGG_API = "https://rest.kegg.jp"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX_API = "https://api.openalex.org"
OPENALEX_EMAIL = CONTACT_EMAIL  # was a dead example.edu literal; now the real contact
INTERPRO_API = "https://www.ebi.ac.uk/interpro/api"
BIORXIV_API = "https://api.biorxiv.org"

# Discovery + full-text sources added in the retrieval-expansion work.
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
CROSSREF_API = "https://api.crossref.org"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "https://export.arxiv.org/api"
# Optional free key (semanticscholar.org).  Keyless works but the shared pool
# 429s after ~a dozen calls; setting this raises the limit.  Zero-code to add.
# NOTE: the documented KEYED default is still 1 req/s (higher only by grant),
# so a key removes the shared-pool contention but not the per-second cap.
S2_API_KEY = os.environ.get("ENZYME_AGENT_S2_API_KEY", "")

# OpenAlex retired the mailto polite pool on 2026-02-13 and now meters by
# credits.  Live API headers (probed Aug 2026) report a KEYLESS budget of
# x-ratelimit-limit=1000 credits/day at 10 credits per list call -- and the
# parallel USD dimension agrees (x-ratelimit-limit-usd=0.1 at 0.001/call) --
# i.e. ~100 searches/day before HTTP 409.  (An early announcement cited a lower
# 100-credit figure; the live API currently reports 1000.)  A key raises it to
# 100,000 credits/day.  Sent as the ``api_key`` query param.
OPENALEX_API_KEY = os.environ.get("ENZYME_AGENT_OPENALEX_API_KEY", "")

# NCBI E-utilities: a free api_key raises the limit from 3 to 10 req/s.  NCBI
# also asks every request to carry ``tool`` and ``email`` identifiers.
NCBI_API_KEY = os.environ.get("ENZYME_AGENT_NCBI_API_KEY", "")
NCBI_TOOL = "enzyme-discovery-agent"

# --- Client-side rate limiting ---
# Proactive per-host minimum interval between requests (seconds), enforced in
# the shared HTTP client.  The agent runs tools in a thread pool, so several
# searches can hit the same host at once; spacing them politely avoids the
# bursts that trip 429s (especially the keyless Semantic Scholar pool).  Hosts
# not listed here are unthrottled (interval 0).  Match on host substring.
HTTP_MIN_INTERVAL_DEFAULT = 0.0
HTTP_RATE_LIMITS = {
    # Keyed default is 1 req/s (higher only by grant); the keyless shared pool
    # is also best kept ~1/s.  Override via env if you hold an elevated grant.
    "api.semanticscholar.org": float(os.environ.get("ENZYME_AGENT_S2_MIN_INTERVAL", "1.0")),
    "api.crossref.org": 0.4,              # polite pool ~3 req/s (adapts via headers)
    "www.ebi.ac.uk": 0.2,                 # Europe PMC ~10 req/s per IP; headroom
    # NCBI allows 3 req/s keyless, 10 req/s with an api_key.
    "eutils.ncbi.nlm.nih.gov": 0.1 if NCBI_API_KEY else 0.34,
    "api.openalex.org": 0.1,             # 100 req/s with a key; daily credits bind
    "api.unpaywall.org": 0.1,
    "api.biorxiv.org": 0.5,               # Cloudflare-fronted; do not hammer
}
