"""Catalytic-feasibility kernel: a structured, reaction-specific specification of
what any candidate enzyme's mechanism must be able to do to catalyze the target.

WHY THIS EXISTS
---------------
Today the agent reasons deeply about the target reaction in Phase 1, but that
reasoning decays into notebook prose while the machinery downstream runs on
heuristic proxies (an EC->bond lookup table for "feasibility", fingerprint
similarity for "substrate fit"). The propagating ``target_profile`` even keeps its
mechanistic slots (``reaction`` / ``enzyme_requirements``) empty.

The feasibility kernel captures that Phase-1 reasoning as a FIRST-CLASS, structured
object that every downstream stage can consult: the disqualifiers gate screening,
the feasibility tests drive the per-candidate reasoned verdict (and therefore
ranking), and the modeling implications parameterize structural modeling. The
kernel poses the per-reaction questions; the mechanism extractor supplies the
per-candidate answers.

FIREWALL
--------
The kernel's *content* is reasoned AT RUNTIME from the user's own query. Nothing
in this module (schema, prompt, comments) encodes a specific answer, family, or
benchmark substrate: the fields are reaction-agnostic and the authoring prompt is
generic ("reason about the target's activation requirement / topology /
disqualifiers"), exactly like the rest of the agent's strategy prompts. Populated
kernels live only in per-run artifacts, never in this source.

This module is deliberately SELF-CONTAINED and UNWIRED: it defines the schema, a
validating normalizer, and the authoring prompt. Consuming the kernel in
screening / evaluation / ranking is a separate, later change, gated on a
benchmark A/B (see ``eval/ab_compare.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = "1.0"

# Reaction-agnostic enums. Kept broad so the kernel fits any chemistry class
# (substitution, hydrolysis, redox, radical, isomerization, ...) without bias
# toward one — the same principle that keeps the mechanism extractor class-agnostic.
DIRECTIONS = frozenset({
    "formation", "cleavage", "transfer", "redox",
    "isomerization", "rearrangement", "other", "unknown",
})
TOPOLOGY_CLASSES = frozenset({
    "free_small_molecule", "carrier_protein_tethered", "polymer",
    "macromolecule", "mixed", "unknown",
})
EVIDENCE_SOURCES = frozenset({
    "mechanism", "ec", "structure", "literature", "annotation",
})
METHODS = frozenset({"cofold", "ml_mm", "unspecified"})
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _str_list(value: object, field: str) -> list[str]:
    if value is None:
        return []
    _require(isinstance(value, list), f"{field} must be a list.")
    out = []
    for item in value:
        _require(isinstance(item, str) and item.strip(),
                 f"{field} entries must be non-empty strings.")
        out.append(item.strip())
    return out


def _normalize_feasibility_tests(value: object) -> list[dict]:
    """Each test is a graded question answerable from a named evidence source."""
    if value is None:
        return []
    _require(isinstance(value, list), "feasibility_tests must be a list.")
    out = []
    for test in value:
        _require(isinstance(test, dict), "each feasibility_test must be an object.")
        question = test.get("question")
        _require(isinstance(question, str) and question.strip(),
                 "each feasibility_test needs a non-empty question.")
        source = test.get("evidence_source", "mechanism")
        _require(source in EVIDENCE_SOURCES,
                 f"evidence_source must be one of {sorted(EVIDENCE_SOURCES)}.")
        disq = test.get("disqualifying_if_failed", False)
        _require(isinstance(disq, bool), "disqualifying_if_failed must be a boolean.")
        out.append({
            "question": question.strip(),
            "evidence_source": source,
            "disqualifying_if_failed": disq,
        })
    return out


def normalize_feasibility_kernel(kernel: dict | None) -> dict:
    """Validate and normalize a feasibility kernel without any target-specific rules.

    Returns a fully-populated kernel dict; raises ``ValueError`` on a malformed one.
    A ``None`` kernel yields an empty, schema-valid skeleton (unknown everywhere) so
    a run that has not yet authored a kernel still type-checks.
    """
    if kernel is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "transformation": {"bond_change": "", "direction": "unknown",
                               "reacting_atoms": None},
            "activation_requirement": "unknown",
            "implied_cofactors_metals": [],
            "catalytic_requirements": [],
            "substrate_topology": {"class": "unknown", "constraint": ""},
            "feasibility_tests": [],
            "disqualifiers": [],
            "modeling_implications": {"cofactors_to_include": [],
                                      "method_first": "unspecified",
                                      "geometry_metric": None},
            "confidence": "low",
            "open_questions": [],
        }
    _require(isinstance(kernel, dict), "feasibility kernel must be a JSON object.")
    _require(kernel.get("schema_version", SCHEMA_VERSION) == SCHEMA_VERSION,
             f"Unsupported kernel schema; expected version {SCHEMA_VERSION}.")

    transformation = kernel.get("transformation") or {}
    _require(isinstance(transformation, dict), "transformation must be an object.")
    direction = transformation.get("direction", "unknown")
    _require(direction in DIRECTIONS, f"direction must be one of {sorted(DIRECTIONS)}.")
    reacting_atoms = transformation.get("reacting_atoms")
    _require(reacting_atoms is None or isinstance(reacting_atoms, str),
             "reacting_atoms must be a string or null.")

    topology = kernel.get("substrate_topology") or {}
    _require(isinstance(topology, dict), "substrate_topology must be an object.")
    topo_class = topology.get("class", "unknown")
    _require(topo_class in TOPOLOGY_CLASSES,
             f"substrate_topology.class must be one of {sorted(TOPOLOGY_CLASSES)}.")

    modeling = kernel.get("modeling_implications") or {}
    _require(isinstance(modeling, dict), "modeling_implications must be an object.")
    method_first = modeling.get("method_first", "unspecified")
    _require(method_first in METHODS, f"method_first must be one of {sorted(METHODS)}.")
    geometry_metric = modeling.get("geometry_metric")
    _require(geometry_metric is None or isinstance(geometry_metric, str),
             "geometry_metric must be a string or null.")

    confidence = kernel.get("confidence", "low")
    _require(confidence in CONFIDENCE_LEVELS,
             f"confidence must be one of {sorted(CONFIDENCE_LEVELS)}.")

    activation = kernel.get("activation_requirement", "unknown")
    _require(isinstance(activation, str) and activation.strip(),
             "activation_requirement must be a non-empty string ('none'/'unknown' allowed).")

    return {
        "schema_version": SCHEMA_VERSION,
        "transformation": {
            "bond_change": str(transformation.get("bond_change", "")).strip(),
            "direction": direction,
            "reacting_atoms": reacting_atoms.strip() if isinstance(reacting_atoms, str) else None,
        },
        "activation_requirement": activation.strip(),
        "implied_cofactors_metals": _str_list(
            kernel.get("implied_cofactors_metals"), "implied_cofactors_metals"),
        "catalytic_requirements": _str_list(
            kernel.get("catalytic_requirements"), "catalytic_requirements"),
        "substrate_topology": {
            "class": topo_class,
            "constraint": str(topology.get("constraint", "")).strip(),
        },
        "feasibility_tests": _normalize_feasibility_tests(kernel.get("feasibility_tests")),
        "disqualifiers": _str_list(kernel.get("disqualifiers"), "disqualifiers"),
        "modeling_implications": {
            "cofactors_to_include": _str_list(
                modeling.get("cofactors_to_include"), "cofactors_to_include"),
            "method_first": method_first,
            "geometry_metric": geometry_metric.strip() if isinstance(geometry_metric, str) else None,
        },
        "confidence": confidence,
        "open_questions": _str_list(kernel.get("open_questions"), "open_questions"),
    }


def load_feasibility_kernel(path: str | Path) -> dict:
    """Load and validate a feasibility kernel from a JSON artifact."""
    return normalize_feasibility_kernel(json.loads(Path(path).read_text()))


# --- Stage-0/1 authoring prompt (generic, firewall-safe) -------------------- #
# Instructs the model to REASON the kernel from the target reaction's chemistry.
# It presupposes no enzyme family and names no answer: the content is generated
# per query at runtime.
KERNEL_AUTHORING_PROMPT = """\
You are reasoning about a target enzymatic reaction to produce a CATALYTIC-FEASIBILITY
KERNEL: a structured specification of what ANY candidate enzyme's catalytic mechanism
must be able to do to catalyze the target. Reason from the chemistry itself. Do NOT
presuppose a specific enzyme family, gene, or answer; do not name one. This works for
any reaction class (substitution, hydrolysis, redox, radical, addition-elimination,
isomerization, ligation, ...).

Return ONLY strict JSON (no prose, no code fences) with this shape:
{
  "transformation": {
    "bond_change": "<one line: which bond forms or breaks>",
    "direction": "formation|cleavage|transfer|redox|isomerization|rearrangement|other|unknown",
    "reacting_atoms": "<the atoms whose approach defines the catalyzed step, or null>"
  },
  "activation_requirement": "<what must happen for the reaction to proceed; 'none' if it is downhill and needs no activated intermediate>",
  "implied_cofactors_metals": ["<cofactor/metal the activation implies, e.g. a phosphoryl donor or a hydride acceptor; [] if none>"],
  "catalytic_requirements": ["<one functional requirement a competent active site must satisfy>", "..."],
  "substrate_topology": {
    "class": "free_small_molecule|carrier_protein_tethered|polymer|macromolecule|mixed|unknown",
    "constraint": "<the topology constraint the target imposes on any competent mechanism>"
  },
  "feasibility_tests": [
    {"question": "<a graded yes/no question to ask of a candidate's mechanism>",
     "evidence_source": "mechanism|ec|structure|literature|annotation",
     "disqualifying_if_failed": true|false}
  ],
  "disqualifiers": ["<a mechanistic CLASS that cannot serve this target (by direction, activation, or topology), described generically>"],
  "modeling_implications": {
    "cofactors_to_include": ["<species to include when modeling a complex; [] if none>"],
    "method_first": "cofold|ml_mm|unspecified",
    "geometry_metric": "<the atom pair whose distance/angle judges a productive pose, or null>"
  },
  "confidence": "high|medium|low",
  "open_questions": ["<what evidence must be gathered to resolve or refine a low-confidence field>"]
}

Rules:
- Ground every field in the reaction's chemistry, not in a presupposed answer.
- If the transformation needs an activated intermediate to proceed, state the
  activation_requirement and the cofactors/metals it implies; otherwise say "none".
- disqualifiers name mechanistic CLASSES (wrong direction, wrong activation, wrong
  substrate topology), never specific enzymes.
- feasibility_tests are the rubric later stages test each candidate against; make
  each answerable from the named evidence source, and mark disqualifying_if_failed
  true only for a genuine hard incompatibility.
- Report only what the chemistry supports. Use "unknown" / "none" / [] and set
  confidence honestly rather than inventing specifics; list what you are unsure of
  in open_questions so downstream evidence can refine the kernel.
"""
