"""Configuration constants for the enzyme discovery agent."""

from __future__ import annotations

import os

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
S2_API_KEY = os.environ.get("ENZYME_AGENT_S2_API_KEY", "")
