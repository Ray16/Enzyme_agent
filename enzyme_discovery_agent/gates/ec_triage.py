"""Gate 2: Auto-screen + auto-evaluate all candidates.

Fires before the first evaluate_candidate call (after entity extraction).
Auto-screens one representative per gene, then auto-evaluates ALL screened
candidates so the LLM reasons over complete data instead of deciding what
to evaluate with limited information.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.chemistry_logic import classify_reaction_chemistry
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.candidate_registry import (
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.models.exploration_log import append_exploration_log
from enzyme_discovery_agent.models.notebook import load_notebook
from enzyme_discovery_agent.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")

def _family_key(metadata: dict, accession: str) -> str:
    """Return a stable, coarse family key without relying on one gene label."""
    genes = metadata.get("gene_names", [])
    if genes:
        return f"gene:{genes[0].lower()}"
    name = re.sub(r"\b\d+\b", "", metadata.get("protein_name", "").lower())
    name = re.sub(r"\s+", " ", name).strip()
    return f"name:{name}" if name else f"accession:{accession}"


def _evidence_tier(metadata: dict) -> int:
    """Order representatives by explicit evidence, never a composite score."""
    if metadata.get("direct_target_evidence") or metadata.get("native_substrates"):
        return 0
    if metadata.get("catalytic_activities") or metadata.get("structural_homology"):
        return 1
    return 2


def auto_screen_candidates(run_dir: Path, log_fn=None) -> None:
    """Auto-screen a capped, diverse set of family representatives.

    Groups all discovered candidates by gene name, picks the best
    representative per gene (prefer longest sequence), and screens
    those that haven't been screened yet.
    """
    from enzyme_discovery_agent.tools.reasoning import _fetch_uniprot_entry_light

    registry = load_candidate_registry(run_dir)
    candidates = registry.get("candidates", {})
    if not candidates:
        return

    family_groups: dict[str, list[tuple[str, dict]]] = {}
    for acc, info in candidates.items():
        meta = info.get("metadata", {})
        family_groups.setdefault(_family_key(meta, acc), []).append((acc, info))

    already_screened = set()
    for acc, info in candidates.items():
        if any(s["status"] in ("screened", "evaluated") for s in info.get("statuses", [])):
            already_screened.add(acc)

    representatives: list[tuple[int, int, str]] = []
    for family, members in family_groups.items():
        # Keep a member if it is known to be full-length (>=100 aa) OR if its
        # length is not known yet.  Full-text-mined accessions are registered
        # WITHOUT a length (only their source); screening is the step that
        # fetches the length, so applying a >=100 cutoff here — before the
        # length exists — silently discards exactly the deterministically
        # recovered enzymes we mined the literature to reach.  Genuine short
        # fragments (a KNOWN length below 100) are still dropped.
        members = [
            (acc, info) for acc, info in members
            if (info.get("metadata", {}).get("seq_length") or 0) >= 100
            or not info.get("metadata", {}).get("seq_length")
        ]
        if not members:
            continue
        members.sort(
            key=lambda x: x[1].get("metadata", {}).get("seq_length", 0),
            reverse=True,
        )
        best_acc = members[0][0]
        if best_acc not in already_screened:
            representatives.append((
                _evidence_tier(members[0][1].get("metadata", {})),
                -members[0][1].get("metadata", {}).get("seq_length", 0),
                best_acc,
            ))

    # No cap: screen EVERY distinct-gene representative.  Orthologs are already
    # collapsed to one rep per gene above, so this is one screen per DISTINCT
    # enzyme, not per raw hit.  Capping here previously dropped whole families
    # before evaluation — and because the tier-2 ("uncurated") bucket is where
    # TrEMBL-annotated (uncurated) candidate enzymes land, the cap systematically
    # discarded the enzymes often closest to the target
    # chemistry.  Relevance filtering belongs at the mechanism-feasibility
    # ranking, never at pre-evaluation screening.
    representatives.sort()
    representatives = [accession for _, _, accession in representatives]

    if not representatives:
        return

    if log_fn:
        log_fn(f"  Auto-screening {len(representatives)} family representatives "
               f"({len(family_groups)} families, {len(already_screened)} already screened)")

    screened_count = 0
    for i in range(0, len(representatives), 10):
        batch = representatives[i:i + 10]
        for acc in batch:
            entry = _fetch_uniprot_entry_light(acc)
            if entry:
                update_candidate_registry(run_dir, acc, "screened", metadata={
                    "protein_name": entry.get("protein_name", ""),
                    "organism": entry.get("organism", ""),
                    "ec_numbers": entry.get("ec_numbers", []),
                    "catalytic_activities": entry.get("catalytic_activities", []),
                    "function": entry.get("function", ""),
                })
                screened_count += 1

    append_exploration_log(
        run_dir,
        entry_type="SCREEN",
        title=f"Auto-screened {screened_count} gene representatives",
        details={
            "Total candidate families": len(family_groups),
            "Already screened": len(already_screened),
            "Newly screened": screened_count,
        },
    )
    if log_fn:
        log_fn(f"  Auto-screening complete: {screened_count} newly screened")


def _extract_target_params(run_dir: Path) -> dict:
    """Load durable target context, with notebook parsing only as legacy fallback."""
    context_path = run_dir / "target_context.json"
    try:
        context = json.loads(context_path.read_text())
        profile = context.get("target_profile", {})
        reaction = profile.get("reaction", {})
        substrate = profile.get("substrate", {})
        reaction_text = " ".join(filter(None, [
            str(reaction.get("class", "")),
            str(reaction.get("direction", "")),
            str(context.get("query", "")),
        ]))
        if reaction_text:
            return {
                "target_ec_class": str(reaction.get("ec_class", "")),
                "target_reaction_type": reaction_text,
                "target_substrate_smiles": str(context.get("substrate_smiles") or ""),
                "target_substrate_description": json.dumps(substrate, sort_keys=True),
            }
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    # Legacy runs written before target_context.json.
    notebook = load_notebook(run_dir)
    reaction_text = notebook.get("sections", {}).get("reaction_analysis", "")

    smiles_match = re.search(r'SMILES[:\s]+([A-Za-z0-9@+\-\[\]\\/()\.\=#]+)', reaction_text)
    smiles_list = []
    for m in re.finditer(r'(?:SMILES|validated)[:\s]+([A-Za-z0-9@+\-\[\]\\/()\.\=#]{3,})', reaction_text):
        s = m.group(1).strip()
        if s and len(s) >= 3:
            smiles_list.append(s)

    ec_match = re.search(r'EC\s*(\d\.\d+(?:\.\d+)?)', reaction_text)

    return {
        "target_ec_class": ec_match.group(1) if ec_match else "",
        "target_reaction_type": reaction_text[:200] if reaction_text else "",
        "target_substrate_smiles": ".".join(smiles_list[:2]) if smiles_list else "",
        "target_substrate_description": "",
    }


def _auto_evaluate_all(run_dir: Path, log_fn=None) -> int:
    """Auto-evaluate ALL screened candidates that haven't been evaluated yet."""
    from enzyme_discovery_agent.tools.reasoning import evaluate_candidate

    registry = load_candidate_registry(run_dir)
    target_params = _extract_target_params(run_dir)

    to_evaluate = []
    for acc, info in registry.get("candidates", {}).items():
        statuses = [s["status"] for s in info.get("statuses", [])]
        if "screened" in statuses and "evaluated" not in statuses:
            to_evaluate.append(acc)

    if not to_evaluate:
        return 0

    if log_fn:
        log_fn(f"  Auto-evaluating {len(to_evaluate)} screened candidates")

    evaluated = 0
    artifacts_dir = run_dir / "artifacts" / "candidates"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    import json as _json
    import threading as _threading
    from datetime import datetime as _dt

    existing = list(artifacts_dir.glob("evaluate_candidate_*.json"))
    _counter = [len(existing)]
    _counter_lock = _threading.Lock()

    def _next_artifact_path():
        with _counter_lock:
            _counter[0] += 1
            return artifacts_dir / f"evaluate_candidate_{_counter[0]:03d}.json"

    def _eval_one(acc):
        try:
            result = evaluate_candidate(
                run_dir=run_dir,
                accession=acc,
                **target_params,
            )
            if result and result.get("status") == "success":
                path = _next_artifact_path()
                with open(path, "w") as f:
                    _json.dump({
                        "tool_name": "evaluate_candidate",
                        "args": {"accession": acc},
                        "timestamp": _dt.utcnow().isoformat(),
                        "iteration": "auto",
                        "result": result,
                    }, f, default=str)
            return acc, result
        except Exception as e:
            logger.warning("Auto-evaluate failed for %s: %s", acc, e)
            return acc, None

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.TOOL_THREAD_POOL_WORKERS
    ) as pool:
        futures = [pool.submit(_eval_one, acc) for acc in to_evaluate]
        for future in concurrent.futures.as_completed(futures):
            acc, result = future.result()
            if result and result.get("status") == "success":
                evaluated += 1

    if log_fn:
        log_fn(f"  Auto-evaluation complete: {evaluated}/{len(to_evaluate)} evaluated")

    append_exploration_log(
        run_dir,
        entry_type="EVAL",
        title=f"Auto-evaluated {evaluated} candidates",
        details={
            "Total screened": len(to_evaluate),
            "Successfully evaluated": evaluated,
        },
    )

    return evaluated


def _check_landscape_coverage(run_dir: Path) -> tuple[bool, list[str]]:
    """Check if the agent has searched for the genes it enumerated in its landscape.

    Returns (is_covered, missing_genes). Coverage is considered sufficient
    when at least 70% of landscape genes have been searched.
    """
    registry = load_candidate_registry(run_dir).get("candidates", {})
    represented_families = {_family_key(info.get("metadata", {}), acc) for acc, info in registry.items()}
    if len(represented_families) >= 3:
        # Focused discovery has sufficient mechanistic breadth; do not force
        # searches for every incidental gene mentioned in a literature note.
        return True, []

    notebook = load_notebook(run_dir)
    landscape = notebook.get("sections", {}).get("enzyme_landscape", "")
    if not landscape:
        return True, []

    gene_re = re.compile(r'\b([A-Z][a-z]{1,3}[A-Z]\d{0,2})\b')
    blacklist = {
        "ATP", "ADP", "AMP", "GTP", "CoA", "NAD", "FAD", "NRPS", "KEGG",
        "PDB", "NCBI", "BLAST", "FPLC", "HPLC", "DMSO", "HEPES",
        "ANL", "NIS", "Fig", "Tab", "Ref",
    }
    landscape_genes = set()
    for m in gene_re.finditer(landscape):
        name = m.group(1)
        if name not in blacklist:
            landscape_genes.add(name.lower())

    if not landscape_genes:
        return True, []

    log_path = run_dir / "exploration_log.md"
    if not log_path.exists():
        return False, sorted(landscape_genes)

    log_text = log_path.read_text()
    searched = set()
    for m in re.finditer(r'gene:(\w+)', log_text):
        searched.add(m.group(1).lower())
    for m in re.finditer(r'query:.*?(\w{3,})', log_text):
        searched.add(m.group(1).lower())

    missing = landscape_genes - searched
    coverage = 1.0 - (len(missing) / len(landscape_genes)) if landscape_genes else 1.0

    return coverage >= 0.7, sorted(missing)


class ECTriageGate:
    """Auto-screen and auto-evaluate all candidates, then present results to LLM."""

    def __init__(self):
        self._coverage_redirected = False

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        # Screening is an explicit candidate-promotion action.  Once one or
        # more candidates exist, evaluation must not depend on the LLM
        # remembering to issue a second, redundant evaluate_candidate call.
        return (
            state.candidate_count > 0
            and not state.eval_gate_fired
            and not state.discovery_coverage_incomplete
        )

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        if not self._coverage_redirected:
            covered, missing = _check_landscape_coverage(state.run_dir)
            if not covered and missing:
                self._coverage_redirected = True
                missing_str = ", ".join(missing[:30])
                logger.info("Redirecting: %d landscape genes not yet searched", len(missing))
                return GateResult(
                    intercept=True,
                    new_messages=[
                        {
                            "role": "user",
                            "content": (
                                f"STOP — you listed {len(missing)} genes in your enzyme "
                                f"landscape that you have NOT searched for yet. Search "
                                f"UniProt for these genes BEFORE evaluating:\n\n"
                                f"{missing_str}\n\n"
                                f"Search each individually using gene_name parameter. "
                                f"Then call evaluate_candidate to trigger evaluation."
                            ),
                        },
                    ],
                    log_message=f"Coverage redirect: {len(missing)} genes unsearched",
                )

        logger.info("Auto-screening all discovered candidates")
        auto_screen_candidates(state.run_dir, log_fn=lambda msg: logger.info(msg))

        logger.info("Auto-evaluating all screened candidates")
        n_evaluated = _auto_evaluate_all(
            state.run_dir, log_fn=lambda msg: logger.info(msg)
        )

        registry = load_candidate_registry(state.run_dir)
        eval_lines = []
        for acc, info in registry.get("candidates", {}).items():
            statuses = [s["status"] for s in info.get("statuses", [])]
            if "evaluated" not in statuses:
                continue
            meta = info.get("metadata", {})
            name = meta.get("protein_name", "unnamed")
            genes = meta.get("gene_names", [])
            gene = genes[0] if genes else ""
            ec_list = meta.get("ec_numbers", [])
            ec_str = ec_list[0] if ec_list else "no EC"
            cat_acts = meta.get("catalytic_activities", [])
            reaction = cat_acts[0][:80] if cat_acts else "no reaction data"
            func = meta.get("function", "")
            func_tag = f" | Function: {func[:60]}" if func else ""

            eval_lines.append(
                f"  - {gene} ({acc}) {name} | EC {ec_str} | {reaction}{func_tag}"
            )

        if not eval_lines:
            return GateResult(
                intercept=False,
                state_updates={"critique_injected": True, "eval_gate_fired": True},
            )

        logger.info("Auto-evaluation complete: %d candidates evaluated", len(eval_lines))

        new_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Auto-screening and auto-evaluation complete. "
                    f"{len(eval_lines)} candidates have been fully evaluated "
                    f"with substrate similarity, literature, and structural data."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"All {len(eval_lines)} screened candidates have been "
                    f"automatically evaluated with full data (substrate "
                    f"similarity, literature support, structural availability). "
                    f"Evaluation data is available for each candidate.\n\n"
                    f"Evaluated candidates:\n"
                    + "\n".join(sorted(eval_lines))
                    + "\n\nProceed directly to RANKING. Review the evaluation "
                    "data for each candidate and rank them according to the "
                    "ranking framework. Do NOT call evaluate_candidate again "
                    "— all candidates have already been evaluated."
                ),
            },
        ]

        return GateResult(
            intercept=True,
            new_messages=new_messages,
            log_message=f"Auto-evaluated {len(eval_lines)} candidates",
            state_updates={"critique_injected": True, "eval_gate_fired": True},
        )
