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
# The auto-screen light-annotation fetch and the auto-evaluation loop are pure
# network I/O (no LLM), and every distinct gene representative is now evaluated,
# so a higher concurrency keeps finalization fast without over-loading the public
# APIs.  Kept modest to stay polite to UniProt / EBI.
EVAL_POOL_WORKERS = 8
# Concurrency for the full-text harvest: papers are downloaded and read (LLM
# extraction) in parallel rather than one at a time.  Bounded to stay polite to
# publisher servers and the LLM endpoint.
FULLTEXT_HARVEST_WORKERS = 6
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
FULLTEXT_HARVEST_MAX_PAPERS = 6
# Full-text mining channels.  The regex DEPOSITED-accession miner (primary papers)
# is always on and near-free — it carries literature's irreplaceable value
# (characterised-but-poorly-databased enzymes that DB/family search miss).  The
# LLM NAMED/CITED-enzyme extraction is expensive and low-precision (most cited
# names resolve to nothing or off-target), so it is OFF by default; enable only
# when name-level recall is worth the token cost.
FULLTEXT_LLM_NAMED_EXTRACTION = False
MAX_RESULT_CHARS = 20_000
# Screening allocates a RELEVANCE BUDGET across fold families, it does not blindly
# cap.  auto_screen collapses orthologs to one representative per distinct gene,
# groups those representatives by their fold family, and then spends a bounded
# evaluation budget where the grounded signal is: EVERY fold keeps at least
# FAMILY_EVAL_FLOOR_PER_FOLD representative (no family is dropped before
# evaluation — the earlier concern that a flat cap discarded whole families, and
# with them the uncurated/TrEMBL bucket closest to the target chemistry), while
# EXTRA depth flows to the folds whose members' native reactions most resemble
# the target (same reaction direction, closer substrate).  A blind per-gene screen
# instead spent budget evenly and let one large off-target fold crowd out the
# on-target ones.  The floor guarantees breadth; the budget concentrates depth.
# Coverage policy: the diversity was already there (one representative per gene,
# spread across fold families); what was missing is DEPTH within the RELEVANT
# families.  A tight per-fold cap (was 6) discarded most function-distinct
# annotated members of an on-target family before ranking, so a family with 30
# same-direction role-complete members contributed only 6 to the report.  The cap
# is now high enough not to bite for a coherent catalytic family (relevant amide
# families here carry <=265 reviewed members, far fewer after ortholog collapse
# to gene representatives), so on-target families are enumerated WIDELY.  Breadth
# is still guaranteed by the floor (every fold keeps >=1), and the budget is spent
# in relevance order, so the extra depth flows to the on-target folds while an
# off-target fold stays at its single audit representative.  Raising the cap is
# purely additive to the ranked tail: ranking still orders by relevance, so the
# top candidates are unchanged; we simply stop dropping lower-ranked but genuine
# annotated members of the relevant families.
# These are SAFETY BOUNDS, not targets: they are set to (or above) the natural
# enumeration ceiling so they never constrain a RELEVANT family below what the
# survey can even enumerate.  A surveyable fold carries at most _LIST_PAGE_MAX
# (500) reviewed members (bigger entries are excluded as superfamilies), so a
# per-fold cap of 500 is provably non-constraining for any relevant fold — not an
# arbitrary number.  The total budget is set to the same ceiling so several
# relevant folds can each expand fully; breadth is still guaranteed by the floor
# (every fold keeps >=1), and depth is still spent in relevance order, so an
# off-target fold stays at its single audit representative while on-target folds
# absorb the depth.  Evaluation is cheap here (UniProt + RDKit + AlphaFold lookup,
# not de novo structure), so evaluating the full relevant set is affordable.
FAMILY_EVAL_BUDGET = 500          # total representatives to evaluate across folds
FAMILY_EVAL_FLOOR_PER_FOLD = 1    # always evaluate at least this many per fold
FAMILY_EVAL_MAX_PER_FOLD = 500    # = enumeration ceiling: never caps a relevant
                                  # fold below its enumerable member count
# Reviewed (manually-curated, high-confidence) members of a relevant fold are
# evaluated in FULL — that is the comprehensive annotated coverage we want.  The
# unreviewed (TrEMBL) tail is lower-confidence and far more numerous, so it is
# bounded SEPARATELY per fold: enough to sample the uncurated-but-annotated
# neighbours of the relevant chemistry, without one fold's thousands of TrEMBL
# entries flooding the ranked report.
FAMILY_TREMBL_EVAL_CAP_PER_FOLD = 50
# Ortholog fan-out cap at DISCOVERY.  A broad family search returns dozens of
# orthologs of the SAME enzyme (one run registered 66 copies of a single gene),
# which crowds out family diversity — the pool re-discovers one enzyme in many
# organisms instead of reaching the next distinct family.  Register only the best
# few representatives per distinct enzyme (family_key); the rest are surplus
# orthologs, trivially re-queried later once an enzyme proves interesting.  The
# goal is efficient exploration of DIFFERENT families, not ortholog breadth.
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
# How many DISTINCT on-target EC reactions of a family get unreviewed (TrEMBL)
# deepening.  This was wrongly coupled to MAX_REPRESENTATIVES_PER_FAMILY (=2, an
# ORTHOLOG cap), which capped a family to 2 reactions' worth of unreviewed members
# even when its reviewed members span more distinct on-target reactions.  Each EC
# is a DISTINCT reaction (function-diverse, not orthologs), so deepening more of
# them widens annotated coverage without ortholog flooding; each pull is still
# bounded by the per-EC page cap (_MEMBER_CAP).  Decoupled and set generously so a
# coherent catalytic family's several on-target reactions are all deepened.
FAMILY_TREMBL_EC_ANCHORS = 8
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
# Prompt-history pruning: the re-sent prompt is ~97% of loop token spend and
# grows as the transcript accumulates.  Three superseded categories are stubbed
# from OLD turns (recent turns kept full so the model always sees fresh context):
#   * tool results   — candidates are in the registry, full data under artifacts/;
#   * assistant TEXT — the action is recorded by its tool_calls (KEPT); only the
#     verbose narration is dropped;
#   * state headers  — each iteration injects a fresh cumulative snapshot, so
#     every earlier one is strictly superseded (keep only the latest — lossless).
# Generous windows by default; tune down for cheaper runs after A/B validation.
HISTORY_KEEP_RECENT_TOOL_MSGS = 16
HISTORY_KEEP_RECENT_ASSISTANT_MSGS = 12
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
