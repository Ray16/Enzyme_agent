"""Configuration constants for the enzyme discovery agent."""

from __future__ import annotations

import os
import json
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
ARGO_MODEL = "argo:claude-5-sonnet"
ARGO_HIGH_REASONING_OUTPUT_RESERVE = 24576
ARGO_HIGH_REASONING_MIN_TIMEOUT = 300

# --- Fallback LLM backend (availability failover) ---
# The agent speaks the OpenAI chat-completions protocol, so any OpenAI-compatible
# endpoint can serve it.  When the primary (argo) is down, quota-revoked, or
# rate-limited, the client fails over (sticky per run) to the next configured
# backend.  Keys come from the ENVIRONMENT ONLY -- never commit a key to the tree.
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
# gemini-3.5-flash-lite free-tier pacing (15 RPM); lower it for other/stricter models.
GEMINI_REQUESTS_PER_MINUTE = float(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "15"))
GEMINI_INPUT_TOKENS_PER_MINUTE = int(os.environ.get("GEMINI_INPUT_TOKENS_PER_MINUTE", "0"))
# Capability priority for text/tool models with nonzero free project quotas.
# This is a routing policy, not a guarantee of ranking on every scientific task.
GEMINI_FREE_MODEL_PRIORITY = (
    "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3-flash-preview",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-2.5-flash", "gemini-2.5-flash-lite",
)


def ordered_gemini_models(value: str | None = None) -> tuple[str, ...]:
    """Select a free-model subset while retaining capability priority."""
    selected = set(GEMINI_FREE_MODEL_PRIORITY if value is None else
                   (item.strip() for item in value.split(",") if item.strip()))
    unknown = selected.difference(GEMINI_FREE_MODEL_PRIORITY)
    if unknown:
        raise ValueError("GEMINI_CASCADE_MODELS contains models outside the free ladder: "
                         + ", ".join(sorted(unknown)))
    if not selected:
        raise ValueError("GEMINI_CASCADE_MODELS must contain at least one free model")
    return tuple(model for model in GEMINI_FREE_MODEL_PRIORITY if model in selected)


GEMINI_CASCADE_MODELS = ordered_gemini_models(os.environ.get("GEMINI_CASCADE_MODELS"))
# Unknown account quotas use a conservative rate, then learn from QuotaFailure.
GEMINI_CASCADE_REQUESTS_PER_MINUTE = float(os.environ.get("GEMINI_CASCADE_REQUESTS_PER_MINUTE", "2"))
GEMINI_MODEL_LIMITS = json.loads(os.environ.get("GEMINI_MODEL_LIMITS", "{}"))
# Leave answer capacity after the high thinking allowance used by Gemini 2.5.
# For newer models this is output headroom, not a fixed thinking-budget request.
GEMINI_HIGH_REASONING_OUTPUT_RESERVE = 24576
GEMINI_HIGH_REASONING_MIN_TIMEOUT = 300
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_ENABLED = os.environ.get("ENZYME_AGENT_GROQ_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}
# Published free-plan limits; keep project credentials outside this package.
GROQ_REQUESTS_PER_MINUTE = int(os.environ.get("GROQ_REQUESTS_PER_MINUTE", "30"))
GROQ_TOKENS_PER_MINUTE = int(os.environ.get("GROQ_TOKENS_PER_MINUTE", "8000"))
GROQ_REQUESTS_PER_DAY = int(os.environ.get("GROQ_REQUESTS_PER_DAY", "1000"))
GROQ_TOKENS_PER_DAY = int(os.environ.get("GROQ_TOKENS_PER_DAY", "200000"))
GROQ_MAX_COMPLETION_TOKENS = int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "4096"))
DEFAULT_MODEL = os.environ.get("ENZYME_AGENT_MODEL") or "claude-code:claude-sonnet-5-5"
ADAPTIVE_REASONING = os.environ.get("ENZYME_AGENT_ADAPTIVE_REASONING", "1") == "1"
ADAPTIVE_REVIEW_MIN_ASSESSED = 20
# Deliberation is the expensive part of an agentic run and retrieval is the cheap part, so
# a review should wrap a substantial batch of evidence work rather than a handful of actions.
# At four actions per cycle a production run spent 19 of 34 calls on review bookkeeping
# (conclude_research_review, interpret_research_outcome, get_exploration_state,
# get_decision_context) against a single database search, and every one of those turns
# re-sends the full context. Overridable per campaign without a code change.
RESEARCH_REVIEW_MAX_INSPECTIONS = int(os.environ.get("ENZYME_REVIEW_MAX_INSPECTIONS", "3"))
RESEARCH_REVIEW_MAX_ACTIONS = int(os.environ.get("ENZYME_REVIEW_MAX_ACTIONS", "12"))
# Google AI Studio replaces the default OpenRouter fallback. The older backend
# remains available only when explicitly selected for an existing workflow.
FALLBACK_PROVIDER = os.environ.get("ENZYME_AGENT_FALLBACK_PROVIDER", "gemini").strip().lower()
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get(
    # Free, tool-calling-capable, Western-hosted (smoke-tested 2026-10-04). Switch to
    # "anthropic/claude-sonnet-5-5" once OpenRouter credits are added for best quality.
    "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
# Optional reasoning allowance for short, non-tool structured extraction calls.
# Leave disabled unless the selected provider/model supports reasoning budgets.
OPENROUTER_STRUCTURED_REASONING_TOKENS = int(os.environ.get(
    "OPENROUTER_STRUCTURED_REASONING_TOKENS", "0"))


def free_model_backends() -> list[dict]:
    """Free text/tool ladder, with optional Groq before the final Google tier."""
    backends = []
    for model in GEMINI_CASCADE_MODELS:
        if model == "gemini-2.5-flash-lite" and GROQ_API_KEY and GROQ_FALLBACK_ENABLED:
            backends.append({"name": "groq", "base_url": GROQ_BASE_URL,
                             "api_key": GROQ_API_KEY, "model": GROQ_MODEL})
        if GEMINI_API_KEY:
            backends.append({"name": "gemini", "base_url": GEMINI_BASE_URL,
                             "api_key": GEMINI_API_KEY, "model": model})
    if (GROQ_API_KEY and GROQ_FALLBACK_ENABLED
            and not any(row["name"] == "groq" for row in backends)):
        backends.append({"name": "groq", "base_url": GROQ_BASE_URL,
                         "api_key": GROQ_API_KEY, "model": GROQ_MODEL})
    return backends


def llm_backends() -> list[dict]:
    """Ordered LLM backends for availability failover (first = primary).

    A backend is included only when configured (endpoint + key present), so an
    unset key omits that provider from automatic routing.
    """
    backends = [{
        "name": "argo",
        "base_url": ARGO_BASE_URL,
        "api_key": ARGO_API_KEY,
        "model": ARGO_MODEL,
    }]
    if FALLBACK_PROVIDER not in {"gemini", "openrouter", "none"}:
        raise ValueError("ENZYME_AGENT_FALLBACK_PROVIDER must be gemini, openrouter, or none")
    if FALLBACK_PROVIDER == "gemini":
        backends.extend(free_model_backends())
    if FALLBACK_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        backends.append({
            "name": "openrouter",
            "base_url": OPENROUTER_BASE_URL,
            "api_key": OPENROUTER_API_KEY,
            "model": OPENROUTER_MODEL,
        })
    return backends
LLM_TIMEOUT = 120  # seconds per LLM call
# Sonnet 5 rejects the legacy OpenAI ``temperature`` parameter.  Leaving it
# unspecified uses the provider default and remains compatible with Argo.
LLM_TEMPERATURE = None

# --- Agent defaults ---
MAX_ITERATIONS = 100  # legacy planning hint; campaign resource budgets control stopping
REFLECTION_INTERVAL = 5  # reflect every N iterations
MAX_AUTOMATED_BATCHES_BEFORE_REFLECTION = 4  # bounded evidence tranche before a model decision; never a scientific stopping rule
TARGET_REVIEW_BATCH_SIZE = 8  # amortize the shared target context across distinct candidates
TARGET_REVIEW_MAX_REQUEST_CHARS = 100000  # reduce batch size, never truncate scientific records
AUTO_TARGET_REVIEW = False  # discovery proposes candidates; detailed assessment is optional
DISCOVERY_PORTFOLIO_SIZE = 48
DISCOVERY_RESEARCH_INTERVAL = 10
LITERATURE_DISCOVERY_PAPERS_PER_STEP = 2
EVALUATIONS_PER_STEP = 20
EXPANSION_SEEDS_PER_STEP = 2
ENABLE_EXPANSION_FEEDBACK = True
DISCOVERY_EXTRACTION_TIMEOUT = 90
FINALIZATION_RESERVE_ITERATIONS = 8  # retained for compatibility; no turn-based cutoff

# --- Tool execution ---
TOOL_THREAD_POOL_WORKERS = 4
# The auto-screen light-annotation fetch and the auto-evaluation loop are pure
# network I/O (no LLM), and every distinct gene representative is now evaluated,
# so a higher concurrency keeps finalization fast without over-loading the public
# APIs.  Kept modest to stay polite to UniProt / EBI.
EVAL_POOL_WORKERS = 8
# Concurrency for the full-text harvest: papers are downloaded and read (LLM
# extraction) in parallel rather than one at a time.  Bounded to stay polite to
# publisher servers and the LLM endpoint.
FULLTEXT_HARVEST_WORKERS = 6
# Concurrency for building a fold's members in survey_family_reactions (each member
# does independent ChEBI/UniProt network work -- the dominant per-fold cost).  The
# family-survey gate already runs folds ~6-wide, so keep this modest to stay polite
# to the EBI/UniProt REST endpoints (total in-flight ~= folds x this).
FAMILY_SURVEY_MEMBER_WORKERS = 6
# Optional cache of mechanism-family derivation across runs, keyed on the target,
# retrieved evidence and serving model. Disabled by default for cold discovery.
# The derivation is keyed on the reaction contract
# (bonds + direction + substrate SMILES).  The derivation samples the LLM to
# convergence over the literature (~7 min in the canonical-run profile) and its
# result is contract-stable, so a repeat run of the same target reuses it.  Cache
# lives under ~/.cache/enzyme_agent (override: ENZYME_AGENT_CACHE_DIR).
ENABLE_MECHANISM_FAMILY_CACHE = os.environ.get(
    "ENZYME_AGENT_MECHANISM_FAMILY_CACHE", "0") == "1"
# How many times to RETRY a thin/empty LLM report with the compact prompt before
# writing the deterministic ranking.  Profiling showed this argo model reliably
# returns unparseable long-form reports and the retries (each a full ~4k-token
# generation) fell back to deterministic anyway -- ~5 min of pure waste -- so the
# default is 0: skip straight to the deterministic report.  Raise if a backend that
# reliably produces long-form Markdown is in use.
REPORT_COMPACT_RETRIES = int(os.environ.get("ENZYME_AGENT_REPORT_COMPACT_RETRIES", "0"))
# Cap on how many papers enter the full-text harvest (the expensive part: one LLM
# relevance-gate call per paper, then download + LLM extraction).  The mechanism-
# seed searches inflate the pool to ~10^2 papers; relevance-gating all of them up
# front spends the run's early effort READING rather than SEARCHING for candidates.
# This is now the number of OPEN-ACCESS full texts the lean harvest READS (it walks
# the relevance-ranked pool and downloads until this many succeed; paywalled papers
# fast-fail and are skipped).  Kept small on purpose: the family → InterPro → survey
# → substrate-fit path is the primary candidate engine, and literature's role here is
# a lean named/cited-enzyme recall over a few high-relevance readable papers — NOT a
# broad paper sweep or a full-text-download-gated bottleneck.  Raise if a specific
# characterised enzyme is being missed.
# Profiling (canonical run, 2026-10-03) showed the harvest cost ~8 min and
# registered 0 candidates (all came from search_uniprot), so this is capped low:
# the per-paper LLM extraction dominates wall-clock and scales with this number.
FULLTEXT_HARVEST_MAX_PAPERS = 3
MAX_RESULT_CHARS = 20_000
# Screening allocates a RELEVANCE BUDGET across fold families, it does not blindly
# cap.  auto_screen groups only exact known sequences, preserves unknown identities,
# groups candidates by their fold family, and then spends a bounded
# evaluation budget where the grounded signal is: EVERY fold keeps at least
# FAMILY_EVAL_FLOOR_PER_FOLD representative (no family is dropped before
# evaluation — the earlier concern that a flat cap discarded whole families, and
# with them the uncurated/TrEMBL bucket closest to the target chemistry), while
# EXTRA depth flows to the folds whose members' native reactions most resemble
# the target (same reaction direction, closer substrate).  A blind per-gene screen
# instead spent budget evenly and let one large off-target fold crowd out the
# on-target ones.  The floor guarantees breadth; the budget concentrates depth.
# Budgets count representatives; exact-sequence grouping is possible only when
# sequences are available, and does not establish functional equivalence.
FAMILY_EVAL_BUDGET = 100          # rolling screening tranche, not a campaign cap
FAMILY_EVAL_FLOOR_PER_FOLD = 1    # always evaluate at least this many per fold
FAMILY_EVAL_MAX_PER_FOLD = 20     # depth per fold in this tranche; retain the rest
# The survey may use this bound when promoting representatives. Discovery itself
# retains all eligible search accessions; a name alone cannot prove redundancy.
MAX_REPRESENTATIVES_PER_FAMILY = 2
EVIDENCE_SELECTION_SEARCH_THRESHOLD = 16
# Substrate-fit member promotion + gated TrEMBL deepening.
#
# The family survey enumerates a fold's members WITH their native substrates and
# scores each member's native-substrate resemblance to the target across every
# physicochemical dimension (functional groups, MW, volume, main-chain length,
# polarity).  The members whose native substrate most resembles the target — never
# IDENTICALLY, since enzymes are promiscuous and a non-natural target is no one's
# exact native substrate — are the relevant enzymes in that family, and are
# PROMOTED to candidates so they enter evaluation instead of only being shown to
# the model.
#
# Deliberately NO tuned relevance threshold: selection is RELATIVE (rank the
# best-fitting same-direction members), and the existing family evaluation budget
# grades them — no magic fit cutoff decides relevance here.  The only bound is how
# many representatives to carry, which REUSES MAX_REPRESENTATIVES_PER_FAMILY rather
# than adding a new knob.  Likewise, TrEMBL deepening anchors on the EC of the
# best-fitting REVIEWED member(s) (relative, not an absolute-fit gate): the fold's
# ~10^5 uncurated members collapse to the handful catalysing the reaction the
# reviewed pass judged closest to the target, pulled one page (UNIPROT_MAX... capped
# at the module member cap).  A feature toggle, not a threshold:
FAMILY_INCLUDE_TREMBL = True
# Budget-conscious protection.  Substrate-fit-promoted members are guaranteed
# evaluation (protected from the fold budget), but protecting the top-2 of EVERY
# surveyed fold force-evaluates low-fit families too (e.g. an ammonia-amine
# synthetase whose native amine is nothing like the target) — spending compute on
# families we would only rank low anyway.  So protect the GLOBALLY best-fit
# promotions across ALL families, not per fold: a resource budget (like the
# FAMILY_EVAL_* budgets), applied by RELATIVE fit rank — no relevance threshold.
# Promotions beyond it are still registered and still eligible via the normal
# relevance-budgeted screen; they just do not force-consume evaluation budget.
# The PRIMARY leak control is upstream: promotion now requires a GROUNDED same-
# direction native reaction (select_relevant_members), which drops the off-target
# (hydrolase/esterase/oxidoreductase) members that leaked in as 'unknown' direction
# and doubled the eval cost on the leak-diagnosis run.  With that gate doing the
# real filtering, this cap is a generous BACKSTOP — set comfortably above where
# genuine on-target hits tend to rank by fit, so protection never drops a real hit.
FAMILY_PROTECT_BUDGET = 12
# Structural (Foldseek) expansion seeds per run.  Kept bounded so Foldseek stays
# a within-family expansion, not another broad search, but 2 was too tight once
# the registry spans many evaluated families — it fanned out from only the top
# two, missing the family neighbours of lower-ranked but distinct seeds.
MAX_STRUCTURAL_EXPANSION_SEEDS = 6
# ESM Metagenomic Atlas target database on the public Foldseek server (standard,
# non-motif metagenomic structure index) used by search_esm_atlas_homologs.
ESM_ATLAS_FOLDSEEK_DB = "mgnify_esm30"
MIN_LITERATURE_SEARCH_ROUNDS = 3
# Family-coverage guard: a run must not finalize while a mechanism family the
# agent itself derived as promising (high/medium) was never targeted by a search.
# Each unexplored family triggers a bounded targeted-search nudge; after this many
# nudges the run may finalize anyway (escape hatch — coverage is encouraged, never
# hard-blocked, so a family whose descriptors resist search can't stall the run).
MAX_FAMILY_COVERAGE_NUDGES = 4

# --- LLM context compaction ---
ABSTRACT_TRUNCATE_CHARS = 200
DESCRIPTION_TRUNCATE_CHARS = 200
PAPER_LIST_CAP = 8
RESULTS_LIST_CAP = 10
# Prompt-history pruning: keep the most recent N tool results at full size and
# stub older ones (their state is durably persisted).  Bounds the re-sent
# prompt, which is ~97% of loop token spend.  Generous by default so the model
# always sees several recent evidence turns in full.
HISTORY_KEEP_RECENT_TOOL_MSGS = 16
MODEL_CONTEXT_MAX_CHARS = 16_000
MODEL_CONTEXT_RECENT_ASSISTANTS = 2
MODEL_CONTEXT_TOOL_RESULT_CHARS = 2000
COMPACT_SYSTEM_PROMPT = True
HISTORY_STUB_OVER_CHARS = 1_000

# --- Ranking ---
# Focus ranking: the LLM reasons over the grounded evidence of only the most
# promising candidates (this thinking model returns empty output when asked to
# reason over more than ~10-12 at once), while the rest are ordered
# deterministically beneath them.
RANKING_FOCUS_SHORTLIST = 10
# Structures are staged only for the top ranked leads (a handoff artifact for
# downstream modelling), never the whole evaluated pool.
STRUCTURE_DOWNLOAD_TOP_N = 10

# --- Literature search ---
OVERFETCH_RATIO = 3
RERANK_MIN_SCORE = 0.25
# Stricter bar for which papers get full-text-mined for sequence accessions:
# mining registers candidates, so it must not pull in irrelevant papers.  This
# is only a coarse pre-filter to avoid fetching obvious off-topic PDFs; content
# relevance is enforced by the deposit-cue window (accession must sit next to
# "deposited/accession/GenBank" text) and by the downstream enzyme filter and
# mechanism ranking.
FULLTEXT_MINE_MIN_SCORE = 0.55
# Keyword-result order is the reliable baseline.  Semantic re-ranking is
# opt-in because its Torch/Transformers import can stall or destabilize an
# otherwise network-bound discovery run.
ENABLE_SEMANTIC_RERANK = os.environ.get("ENZYME_AGENT_ENABLE_SEMANTIC_RERANK", "0") == "1"

# --- Deep literature search (citation-graph snowball) ---
# Per-run caps on how many papers the citation snowball pulls, so the deduped pool
# (fed to regex entity extraction) stays bounded. OpenAlex is the primary graph;
# the extra graphs (Semantic Scholar / Europe PMC / PubMed) each add coverage +
# redundancy up to their own cap (backward references + forward citations combined).
DEEP_SEARCH_BACKWARD_CAP = 60      # OpenAlex backward references, total
DEEP_SEARCH_FORWARD_CAP = 40       # OpenAlex forward citations, total
DEEP_SEARCH_EXTRA_SOURCE_CAP = 40  # per extra graph, backward + forward combined

# --- Candidate embedding map (agent-invoked ESM neighbor exploration) ---
# Opt-in: the heavy torch/transformers embedding runs only in a GPU subprocess
# (tools/esm_embed_worker.py) launched via the cluster reservation gate, so a
# normal run never imports torch or touches a GPU. The map_candidate_neighbors
# tool is registered regardless but no-ops when this flag is off.
ENABLE_CANDIDATE_MAP = os.environ.get("ENZYME_AGENT_ENABLE_CANDIDATE_MAP", "0") == "1"
CANDIDATE_MAP_ESM_MODEL = os.environ.get(
    "ENZYME_AGENT_ESM_MODEL", "facebook/esm2_t33_650M_UR50D"
)
CANDIDATE_MAP_STEER_MODE = "exploit"  # future hook: "explore" / "balanced"
# Fallback DEFAULTS only -- the agent overrides these per call via tool params.
CANDIDATE_MAP_TOP_K = 5
CANDIDATE_MAP_PROMISING_N = 10
# GPU embedding subprocess plumbing (never sets CUDA_VISIBLE_DEVICES directly).
CANDIDATE_MAP_GPU_RESERVE = os.environ.get(
    "ENZYME_AGENT_GPU_RESERVE", "/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve"
)
# Dedicated venv with fair-esm layered on palm's CUDA torch (palm's transformers
# is version-broken); loads ESM2 via torch.hub, no transformers/huggingface_hub.
CANDIDATE_MAP_EMBED_PYTHON = os.environ.get(
    "ENZYME_AGENT_EMBED_PYTHON", "/nfs/lambda_stor_01/homes/rzhu/esm_embed_env/bin/python"
)
CANDIDATE_MAP_EMBED_TIMEOUT = 1800  # seconds for the GPU embedding subprocess
# --- Embedding-driven expansion (expand_candidates_esm) ---
# The expansion tool reaches OUTSIDE the discovered pool: it fetches the wider
# membership of the families its own top candidates belong to, embeds them in the
# same representation space, and keeps the ones NEAREST the top candidates as new
# leads. All values are fallback defaults the agent can override per call.
CANDIDATE_MAP_EXPANSION_N_SEEDS = 5           # top candidates to expand around
CANDIDATE_MAP_EXPANSION_K = 10                # new leads kept per seed
CANDIDATE_MAP_EXPANSION_BACKGROUND_PER_FAMILY = 300  # members fetched per family
CANDIDATE_MAP_EXPANSION_MAX_BACKGROUND = 1000        # total background cap (dedup)
CANDIDATE_MAP_EXPANSION_MIN_REVIEWED = 30     # below this, widen fetch to TrEMBL
# Reference enzymes to STAR in the standalone plot are supplied out-of-band
# (CLI/env/optional file), never hardcoded -- the benchmark answer must not live
# in package source (see CLAUDE.md data-leakage firewall). Empty by default.
CANDIDATE_MAP_REFERENCE_ACCESSIONS = os.environ.get("ENZYME_AGENT_MAP_REFERENCE_ACCS", "")

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
# 100-credit figure; the live API currently reports 1000.)  A free key raises
# this ~10x -- observed x-ratelimit-limit=10000 credits/day (~1000 searches),
# not the 100000 some docs claim.  Sent as the ``api_key`` query param.
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
    "api.openalex.org": 0.1,             # 10 req/s baseline; daily credits are the real cap
    "api.unpaywall.org": 0.1,
    "api.biorxiv.org": 0.5,               # Cloudflare-fronted; do not hammer
}

# --- De novo enzyme design (AlphaProtein Novo) ---
# AP Novo is a motif-scaffolding diffusion pipeline (DeepMind, 2026): it
# co-generates backbone + sequence around a fixed catalytic motif, optionally
# redesigns the sequence with LigandMPNN, folds with AlphaFold 3 (leaving-atom
# checkpoint) and scores catalytic-geometry recovery.  The heavy JAX/CUDA stack
# lives in its OWN venv outside this repo, invoked as a subprocess through the
# cluster reservation gate -- a normal agent run never imports jax.
#
# Install layout (code + weights deliberately out of the git tree):
#   $APNOVO_DIR/.venv                     py3.12 + jax[cuda12] + alphafold3
#   $APNOVO_DIR/models/apnovo_generator   generator.bin.zst   (522 MiB)
#   $APNOVO_DIR/models/af3_la             af3_leaving_atom.bin.zst (973 MiB)
APNOVO_DIR = os.environ.get(
    "ENZYME_AGENT_APNOVO_DIR", "/nfs/lambda_stor_01/homes/rzhu/Software/alphaprotein_novo"
)
APNOVO_PYTHON = os.environ.get(
    "ENZYME_AGENT_APNOVO_PYTHON", str(Path(APNOVO_DIR) / ".venv" / "bin" / "python")
)
APNOVO_GENERATOR_MODEL_DIR = os.environ.get(
    "ENZYME_AGENT_APNOVO_GENERATOR_DIR", str(Path(APNOVO_DIR) / "models" / "apnovo_generator")
)
# AF3-LA is fine-tuned for leaving-atom handling, so it models the covalent
# intermediates common in AP Novo designs better than stock AF3.
APNOVO_AF3_MODEL_DIR = os.environ.get(
    "ENZYME_AGENT_APNOVO_AF3_DIR", str(Path(APNOVO_DIR) / "models" / "af3_la")
)
# Optional stage 2 (sequence redesign).  Runs on CPU in a separate py3.11 venv.
LIGANDMPNN_DIR = os.environ.get(
    "ENZYME_AGENT_LIGANDMPNN_DIR", "/nfs/lambda_stor_01/homes/rzhu/Software/LigandMPNN"
)
LIGANDMPNN_PYTHON = os.environ.get(
    "ENZYME_AGENT_LIGANDMPNN_PYTHON",
    str(Path(LIGANDMPNN_DIR) / ".venv-ligandmpnn" / "bin" / "python"),
)
# Fallback DEFAULTS only -- the agent overrides these per call via tool params.
APNOVO_NUM_DESIGNS = 2
APNOVO_SAMPLING_STEPS = 1000        # full-length trajectory; 50 is the smoke setting
APNOVO_MAX_GPUS = 8                 # one node's worth; AP Novo shards designs 1 worker/GPU
APNOVO_TIMEOUT = int(os.environ.get("ENZYME_AGENT_APNOVO_TIMEOUT", str(12 * 3600)))
# Cap BLAS/OMP threads per worker so N GPU workers don't oversubscribe a shared node.
APNOVO_THREADS_PER_WORKER = 4
# Problem-specific metric suites shipped with AP Novo (evaluation.suite).
APNOVO_EVAL_SUITES = (
    "kemp_eliminase",
    "serine_esterase",
    "dehp_esterase",
    "carbene_transfer",
    "nitrene_transfer",
)
# AF3 folding (stage 3) routes attention through tokamax kernels that require
# compute capability >= 8.0 (Ampere).  Measured on this cluster: stage 3 dies on
# a V100 (sm70) with "NotImplementedError: Not supported on Tesla V100-SXM2-32GB"
# -- so folding/evaluation need lambda13 (H100, sm90), while generation (stage 1)
# and the CPU stages run fine on lambda0-6 (V100 sm70 / T4 sm75).
APNOVO_MIN_FOLDING_CAPABILITY = 8.0

# ---------------------------------------------------------------------------
# DISCO -- ligand-conditioned sequence-structure co-design (the motif-free
# generative backend; complements AP Novo's motif scaffolding).
# ---------------------------------------------------------------------------
# DISCO (DIffusion for Sequence-structure CO-design) jointly generates a protein
# SEQUENCE and 3D BACKBONE conditioned on a small molecule / nucleic acid, with
# NO pre-specified catalytic motif -- you hand it only the ligand(s) and a chain
# length.  Where AP Novo needs a 3D arrangement of catalytic sidechain atoms,
# DISCO needs only the species the pocket should complement, so it covers the
# case where the chemistry is known but a trustworthy catalytic geometry is not.
#
# As with AP Novo the heavy torch/CUDA stack lives in its OWN uv venv outside
# this git tree and is invoked as a subprocess through the cluster reservation
# gate, so a normal agent run never imports torch or touches a GPU.
#
# Install layout (code + weights deliberately out of the git tree):
#   $DISCO_DIR/.venv                     uv env: torch(cu13) + deepspeed + disco
#   $DISCO_DIR/models/DISCO.pt           model checkpoint (auto-downloads from HF
#                                        DISCO-Design/DISCO if left unset)
DISCO_DIR = os.environ.get(
    "ENZYME_AGENT_DISCO_DIR", "/nfs/lambda_stor_01/homes/rzhu/Software/DISCO"
)
DISCO_PYTHON = os.environ.get(
    "ENZYME_AGENT_DISCO_PYTHON", str(Path(DISCO_DIR) / ".venv" / "bin" / "python")
)
# Explicit checkpoint path.  Empty string -> let the runner pull it from the
# HuggingFace Hub (DISCO-Design/DISCO) into its cache on first use.
DISCO_CHECKPOINT = os.environ.get(
    "ENZYME_AGENT_DISCO_CHECKPOINT", str(Path(DISCO_DIR) / "models" / "DISCO.pt")
)
# DeepSpeed EvoformerAttention is memory-efficient but compiles CUTLASS kernels
# that need nvcc + a CUTLASS checkout AND an Ampere+ GPU.  Set this to a CUTLASS
# source dir to enable it; left empty, DISCO runs with naive attention (correct,
# but materialises the full attention matrix -> much more memory, so keep lengths
# modest on a 32 GB V100).
DISCO_CUTLASS_PATH = os.environ.get(
    "ENZYME_AGENT_DISCO_CUTLASS_PATH", "/nfs/lambda_stor_01/homes/rzhu/Software/cutlass"
)
# Fallback DEFAULTS only -- the agent overrides these per call via tool params.
DISCO_NUM_SEEDS = 4                 # samples generated per job (one per seed)
DISCO_EXPERIMENT = "diverse"        # the paper's ligand-conditioned preset
DISCO_EFFORT = "max"                # README: conditional generation needs effort=max
DISCO_DEFAULT_LENGTHS = (150, 200, 250)   # exploratory sizes; explicit caller lengths take precedence
DISCO_MAX_GPUS = 1                  # the runner's Fabric is single-process here; 1 GPU/call
DISCO_TIMEOUT = int(os.environ.get("ENZYME_AGENT_DISCO_TIMEOUT", str(12 * 3600)))
DISCO_THREADS_PER_WORKER = 4
# EvoformerAttention/CUTLASS kernels need compute capability >= 8.0 (Ampere).
# Below that DISCO must run with use_deepspeed_evo_attention=false; the tool
# auto-selects based on the host's GPUs and whether DISCO_CUTLASS_PATH is set.
DISCO_MIN_DS_ATTENTION_CAPABILITY = 8.0
