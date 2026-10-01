#!/usr/bin/env python3
"""Resolve a discovery-agent handoff's ``unresolved`` modeling fields.

The discovery agent emits a schema-2.0 handoff in which the mechanistic parts
(activation mode, nucleotide cofactor) are resolved from UniProt but the
setup-dependent choices are left ``unresolved``: enzyme chains, substrate roles
and representation, catalytic-metal stoichiometry, and reaction geometry. Those
are exactly the fields a human previously hand-engineered before running Boltz.

This compiler closes that gap from two grounded sources, keeping both as
provenance:

1. **Literature knowledge** (``literature.py``): PDF-extracted candidate rules,
   matched to the candidate's features and attached as ``literature_basis``
   (paper + claim + quote) on each decision they inform.
2. **Per-target facts**: catalytic-metal stoichiometry derived from UniProt via
   ``scripts/cofactor_stoichiometry.derive_metal_cofactor`` (fixes the blanket
   2xMg -> per-enzyme count), and RDKit re-verification of any changed SMILES.

It applies the pipeline's outcome-grounded **scoring policy guard**: even when a
source paper used pocket restraints, the resolved input carries no ``constraints``
block, because ``CANDIDATE_RANKING_POLICY.md`` scores on unrestrained cofolds and
``angle_gate_proxy.py`` shows restraints invert the signal. Geometry is recorded
as a post-hoc measurement target, not an imposed restraint.

The output is a schema-2.0 handoff whose resolved candidates pass the renderer's
own gates (``build_step1_inputs.execution_shape_errors`` / ``readiness_errors``),
so ``build_step1_inputs.py`` accepts it without ``--allow-unreviewed-exploratory``.

Run under the base env (RDKit): ``~/miniforge3/bin/python``.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

from chemistry import (
    characterize_smiles,
    resolve_cofactor_atom,
    resolve_ligand_atom,
    substrate_errors,
)
from literature import DEFAULT_KNOWLEDGE, LiteratureKnowledge
from reaction import (
    UBIQUITOUS_SPECIES,
    ReactionCoordinates,
    cofactor_species_role,
    load_atom_selectors,
    modeled_species_roles,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # structural_modeling/
SCRIPTS = os.path.join(ROOT, "scripts")


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# The renderer's gates are the real acceptance contract; import them directly so
# a resolved handoff is validated against exactly what will consume it.
_bsi = _load_module("build_step1_inputs", os.path.join(SCRIPTS, "build_step1_inputs.py"))
_cof = _load_module("cofactor_stoichiometry", os.path.join(SCRIPTS, "cofactor_stoichiometry.py"))
execution_shape_errors = _bsi.execution_shape_errors
readiness_errors = _bsi.readiness_errors
derive_metal_cofactor = _cof.derive_metal_cofactor
_first_divalent = _cof._first_divalent
_default_fetch = _cof._fetch_json  # real UniProt fetcher; used when no override is given

FAMILY_METAL = os.path.join(HERE, "knowledge", "family_metal.json")


def _load_family_metal(path: str = FAMILY_METAL) -> dict:
    with open(path) as handle:
        return json.load(handle)


# Which decision points each resolved field draws literature guidance from.
_FIELD_DECISION_POINTS = {
    "enzyme_state": ["participants"],
    "substrates": ["substrate-representation", "participants"],
    "catalytic_metal": ["participants"],
    "geometry": ["pose-or-conformer-selection-criterion"],
    "sampling": ["conformational-sampling-strategy"],
    "validation": ["validation-strategy"],
}


def _candidate_family_ids(candidate: dict) -> list[str]:
    """InterPro family/domain accessions for the candidate, for family-based mechanism transfer."""
    evidence = candidate.get("evidence") or {}
    ids: list[str] = []
    fam = (evidence.get("family_assignment") or {}).get("interpro_family_id")
    if fam:
        ids.append(fam)
    for dom in evidence.get("domain_architecture") or []:
        acc = dom.get("accession")
        if acc:
            ids.append(acc)
    return list(dict.fromkeys(ids))


def _derive_features(candidate: dict, manifest: dict) -> dict[str, Any]:
    """Feature flags used to guard literature-rule applicability.

    Manifest-declared features win; the rest are derived conservatively from the
    handoff so guidance is only attached when its precondition truly holds.
    """
    mi = candidate.get("modeling_input") or {}
    derived = {
        "non_protein_participant": bool(mi.get("target_substrates") or mi.get("cofactors")),
        "task_has_geometric_success_criteria": (mi.get("activation") or {}).get("status") == "assigned",
        "model_is_stochastic": True,
        "binding_context_may_be_multicomponent": bool(mi.get("cofactors")),
        "confidence_metric_used": True,
        "claim_requires_energetics_or_population": False,
    }
    derived.update(manifest.get("features") or {})
    return derived


def _basis(knowledge, field: str, features: dict, extra_points=()) -> list[dict]:
    """Literature basis for a field.

    Draws from the static field map AND any decision point the manifest directive
    names. The latter is what makes the module self-evolving: when newly extracted
    papers add rules under a new decision point, a manifest that references it picks
    them up with no code change.
    """
    points = list(dict.fromkeys(list(_FIELD_DECISION_POINTS.get(field, [])) + [p for p in extra_points if p]))
    blocks: list[dict] = []
    seen: set = set()
    for point in points:
        for block in knowledge.basis_for(point, features):
            key = (point, tuple(block.get("papers") or []), block.get("guidance"))
            if key in seen:
                continue
            seen.add(key)
            blocks.append(block)
    return blocks


def _provenance(directive: dict, literature_basis: list[dict]) -> dict:
    """A per-field resolution record: where the value came from and why."""
    return {
        "origin": directive.get("origin", "inference"),
        "rationale": directive.get("rationale"),
        "evidence_ids": directive.get("evidence_ids") or [],
        "decision_point": directive.get("decision_point"),
        "confidence": directive.get("confidence"),
        "literature_basis": literature_basis,
    }


def _resolve_enzyme_state(mi: dict, manifest: dict, knowledge, features: dict) -> dict | None:
    directive = (manifest.get("resolutions") or {}).get("enzyme_state")
    if not directive:
        return None
    chains = directive.get("chains") or [{"source": "full_sequence"}]
    copies = int(directive.get("copies", 1) or 1)
    resolved_chains: list[dict] = []
    for _ in range(copies):
        resolved_chains.extend(copy.deepcopy(chains))
    mi.setdefault("enzyme_state", {})
    mi["enzyme_state"]["state"] = directive.get("state", mi["enzyme_state"].get("state", "specified"))
    mi["enzyme_state"]["chains"] = resolved_chains
    prov = _provenance(directive, _basis(knowledge, "enzyme_state", features, [directive.get("decision_point")]))
    mi["enzyme_state"]["resolution"] = prov
    return prov


def _resolve_substrates(mi: dict, manifest: dict, knowledge, features: dict) -> tuple[list[dict], list[str]]:
    directives = (manifest.get("resolutions") or {}).get("substrates") or []
    errors: list[str] = []
    provs: list[dict] = []
    substrates = mi.get("target_substrates") or []
    for directive in directives:
        match = directive.get("match") or {}
        target = _match_substrate(substrates, match)
        if target is None:
            errors.append(f"no target substrate matches {match!r}")
            continue
        if "include_in_model" in directive:
            target["include_in_model"] = bool(directive["include_in_model"])
        for key in ("role", "reactive_atom", "name"):
            if directive.get(key) is not None:
                target[key] = directive[key]
        if directive.get("affinity_target") is not None:
            target["affinity_target"] = bool(directive["affinity_target"])
        new_smiles = directive.get("modeling_smiles")
        if new_smiles:
            target["modeling_smiles"] = new_smiles
            target["modeling_representation"] = characterize_smiles(new_smiles)
        if target.get("include_in_model", True):
            errors.extend(
                f"component {target.get('component_index')}: {msg}"
                for msg in substrate_errors(
                    target.get("modeling_smiles"), target.get("modeling_representation") or {}
                )
            )
        # A representation change must be marked reviewed to clear the renderer's gate.
        target["review_status"] = directive.get("review_status", target.get("review_status", "reviewed"))
        prov = _provenance(directive, _basis(knowledge, "substrates", features, [directive.get("decision_point")]))
        prov["component_index"] = target.get("component_index")
        target["resolution"] = prov
        provs.append(prov)
    return provs, errors


def _match_substrate(substrates: list[dict], match: dict) -> dict | None:
    for substrate in substrates:
        if "component_index" in match and substrate.get("component_index") == match["component_index"]:
            return substrate
        if "source_smiles" in match and substrate.get("source_smiles") == match["source_smiles"]:
            return substrate
    return None


def _family_per_site(candidate: dict, family_metal: dict) -> dict:
    """Per-active-site catalytic metal, reasoned from the candidate's mechanism/family.

    Keyed on the handoff's activation mode (acyl_adenylate -> 1 Mg, acyl_phosphate
    -> 2 Mg), falling back to the conservative default. This is the reasoning that
    replaces naive metal counting.
    """
    mode = ((candidate.get("modeling_input") or {}).get("activation") or {}).get("activation_mode")
    entry = (family_metal.get("by_activation_mode") or {}).get(mode) or family_metal.get("default") or {}
    return {
        "per_site_metal_count": int(entry.get("per_site_metal_count", 1)),
        "metal": entry.get("metal", "MG"),
        "rationale": entry.get("rationale"),
        "refs": entry.get("refs") or [],
        "keyed_on": mode or "default",
    }


def _active_sites(mi: dict, directive: dict) -> tuple[int, str]:
    """Number of catalytic sites modeled. Defaults to one per catalytic chain.

    Overridable via the manifest because active sites are not always one-per-chain
    (some ligases have inter-subunit sites)."""
    if isinstance(directive.get("active_sites"), int):
        return int(directive["active_sites"]), "manifest_override"
    chains = ((mi.get("enzyme_state") or {}).get("chains")) or []
    return max(1, len(chains)), "one_per_catalytic_chain"


def _resolve_metal(
    mi: dict, candidate: dict, manifest: dict, knowledge, features: dict, fetch, family_metal: dict
) -> tuple[dict | None, list[str]]:
    directive = (manifest.get("resolutions") or {}).get("catalytic_metal")
    if not directive:
        return None, []
    strategy = directive.get("strategy", "family_reason")
    accession = candidate.get("accession")
    lit = _basis(knowledge, "catalytic_metal", features, [directive.get("decision_point")])

    if strategy == "omit":
        return _provenance(directive, lit), []

    if strategy == "explicit":
        per_site = int(directive.get("per_site_metal_count", directive.get("count", 1)) or 1)
        ccd = directive.get("ccd", "MG")
        family_basis = {"source_type": "manifest_explicit"}
        confidence = directive.get("confidence", "medium")
    elif strategy == "family_reason":
        family = _family_per_site(candidate, family_metal)
        # A manifest can override the per-site count; else family reasoning decides.
        per_site = int(directive.get("per_site_metal_count", family["per_site_metal_count"]))
        evidence_names = ((candidate.get("evidence") or {}).get("cofactor_annotations") or {}).get("names") or []
        ccd = directive.get("ccd") or _first_divalent(evidence_names) or family["metal"]
        family_basis = {
            "source_type": "family_mechanism_reasoning",
            "keyed_on": family["keyed_on"],
            "rationale": family["rationale"],
            "refs": family["refs"],
        }
        confidence = "high" if family["keyed_on"] != "default" else "low"
    else:
        return None, [f"unknown catalytic_metal strategy {strategy!r}"]

    n_sites, sites_basis = _active_sites(mi, directive)
    total = per_site * n_sites

    # UniProt tally: cross-check ONLY. Never the placement count (multimer / label artifacts).
    reference = None
    if directive.get("uniprot_cross_check", True):
        ref = derive_metal_cofactor(accession, family_query=directive.get("family_query"), fetch=fetch or _default_fetch)
        reference = {
            "uniprot_count": int(ref["count"]),
            "uniprot_basis": ref.get("basis"),
            "note": "per-annotation reference only; may conflate oligomeric state and binding-site labels — not authoritative for placement",
            "agrees_with_per_site": int(ref["count"]) == per_site,
        }

    cofactor = {
        "name": ccd,
        "ccd": ccd,
        "count": total,
        "role": directive.get("role", "catalytic_metal"),
        "basis": {
            "derivation_rule": "per_site_metal x active_sites",
            "per_site_metal_count": per_site,
            "active_sites": n_sites,
            "active_sites_basis": sites_basis,
            "family_reasoning": family_basis,
            "uniprot_reference": reference,
            "accession": accession,
        },
        "confidence": confidence,
    }

    cofactors = mi.setdefault("cofactors", [])
    identities = {(c.get("ccd"), c.get("role")) for c in cofactors}
    if (cofactor["ccd"], cofactor["role"]) not in identities:
        cofactors.append(cofactor)
    prov = _provenance(directive, lit)
    prov["resolved_cofactor"] = {
        "ccd": ccd,
        "count": total,
        "per_site_metal_count": per_site,
        "active_sites": n_sites,
        "uniprot_reference": reference,
    }
    return prov, []


def _find_substrate(mi: dict, role: str) -> dict | None:
    for sub in mi.get("target_substrates") or []:
        if sub.get("role") == role and sub.get("include_in_model", True):
            return sub
    return None


def _resolve_attack_atom(mi: dict, spec: dict, nucleotide_ccd: str | None, selectors: dict) -> tuple[dict, list[str]]:
    """Resolve one attack participant (nucleophile/electrophile/leaving group) onto real atoms.

    Step-parameterized and generic: the role/atom_selector come from the reaction
    coordinate; cofactor roles resolve to CCD atom names, ligand roles resolve by
    SMARTS. Abstains (returns an error) when the atom cannot be located.
    """
    role = spec.get("role")
    selector = spec.get("atom_selector")
    if role == "NTP" or role == "catalytic_metal":
        ccd = nucleotide_ccd if role == "NTP" else None
        atom = resolve_cofactor_atom(ccd, selector, selectors.get("cofactor_ccd_atoms") or {})
        resolved = {"kind": "cofactor", "role": role, "ccd": ccd, "atom_selector": selector, "atom": atom}
        errors = [] if atom else [f"cannot resolve {selector} on cofactor {ccd or role}"]
        return resolved, errors
    # ligand participant: locate the modeled substrate carrying this role, match SMARTS
    substrate = _find_substrate(mi, role)
    smarts = (selectors.get("ligand_smarts") or {}).get(selector)
    if substrate is None:
        return {"kind": "ligand", "role": role, "atom_selector": selector, "match_count": 0}, [
            f"no modeled substrate has role {role!r} for the declared step"
        ]
    match = resolve_ligand_atom(substrate.get("modeling_smiles"), smarts)
    resolved = {
        "kind": "ligand",
        "role": role,
        "component_index": substrate.get("component_index"),
        "atom_selector": selector,
        "match_count": match["match_count"],
        "reactive_atom_indices": match["reactive_atom_indices"],
    }
    if not match["match_count"]:
        errors = [f"cannot locate {selector} on {role} ligand"]
    elif match["match_count"] > 1:
        # Fail closed on ambiguity. A selector matching several sites means the
        # downstream consumer silently picks one, and picking the wrong carbonyl
        # scores a different reaction than the one being modelled.
        errors = [
            f"selector {selector!r} is ambiguous on {role}: {match['match_count']} "
            f"matches at atoms {match['reactive_atom_indices']}; tighten the SMARTS "
            f"so it identifies the reacting site uniquely"
        ]
    else:
        errors = []
    return resolved, errors


def _resolve_geometry_for_step(
    mi: dict, manifest: dict, knowledge, features: dict,
    step_spec: dict, mech_key: str, nucleotide_ccd: str | None, selectors: dict,
) -> tuple[dict | None, list[str]]:
    """Derive the near-attack geometry from the DECLARED step, not from activation."""
    directive = (manifest.get("resolutions") or {}).get("geometry") or {}
    attack = step_spec.get("attack") or {}
    errors: list[str] = []
    geometry: dict = {
        "kind": "near_attack",
        "measurement_only": True,  # scoring-policy guard: measured post-hoc, never restrained
        "step": step_spec.get("index"),
        "reactive_state": step_spec.get("reactive_state"),
        "basis": {
            "source_type": "reaction_coordinate",
            "mechanism": mech_key,
            "step_name": step_spec.get("name"),
            "provenance": step_spec.get("provenance"),
        },
    }
    for participant in ("nucleophile", "electrophile", "leaving_group"):
        spec = attack.get(participant)
        if not spec:
            continue
        resolved, atom_errors = _resolve_attack_atom(mi, spec, nucleotide_ccd, selectors)
        geometry[participant] = resolved
        errors.extend(atom_errors)
    mi["geometry"] = geometry
    mi["geometry_status"] = "ready" if not errors else "unresolved"
    prov = _provenance(directive, _basis(knowledge, "geometry", features, [directive.get("decision_point")]))
    prov["derived_from_step"] = {"mechanism": mech_key, "step": step_spec.get("index"), "name": step_spec.get("name")}
    return prov, errors


def _apply_step_species(mi: dict, step_spec: dict, nucleotide_ccd: str | None) -> list[dict]:
    """Keep only cofactors whose species role belongs to this step (drops consumed ones).

    This is what removes stale ATP when modeling step 2: ATP (NTP) is present at
    step 1 but has been consumed into the intermediate by step 2, so it is not in
    step 2's ``species_present`` and is dropped."""
    allowed = set(step_spec.get("species_present") or [])
    kept, dropped = [], []
    for cof in mi.get("cofactors") or []:
        role = cofactor_species_role(cof.get("ccd"), cof.get("role"), nucleotide_ccd)
        if role is None or role in UBIQUITOUS_SPECIES or role in allowed:
            kept.append(cof)
        else:
            dropped.append({"ccd": cof.get("ccd"), "species_role": role, "reason": f"not present at step {step_spec.get('index')}"})
    mi["cofactors"] = kept
    return dropped


def _step_consistency_errors(mi: dict, step_spec: dict, nucleotide_ccd: str | None) -> list[str]:
    """Fail-closed gate: the modeled system must match the declared step.

    This is the safety net that prevents silently modeling the wrong step."""
    errors: list[str] = []
    modeled = modeled_species_roles(mi, nucleotide_ccd)
    required = set(step_spec.get("species_present") or [])
    extra = modeled - required - UBIQUITOUS_SPECIES
    missing = required - modeled - UBIQUITOUS_SPECIES
    for role in sorted(extra):
        errors.append(f"species {role!r} is modeled but not part of step {step_spec.get('index')} ({step_spec.get('name')})")
    for role in sorted(missing):
        errors.append(f"step {step_spec.get('index')} ({step_spec.get('name')}) requires {role!r} but it is not modeled")
    # geometry must describe THIS step's attack, not another step's
    attack = step_spec.get("attack") or {}
    geometry = mi.get("geometry") or {}
    for participant in ("nucleophile", "electrophile"):
        want = (attack.get(participant) or {}).get("role")
        got = (geometry.get(participant) or {}).get("role")
        if want and got != want:
            errors.append(f"geometry {participant} is {got!r} but step {step_spec.get('index')} requires {want!r}")
    return errors


def resolve_candidate(
    candidate: dict,
    manifest: dict,
    knowledge: LiteratureKnowledge,
    *,
    step: int | None,
    fetch: Callable | None = None,
    family_metal: dict | None = None,
    coordinates: ReactionCoordinates | None = None,
    atom_selectors: dict | None = None,
    mechanism_override: str | None = None,
) -> dict:
    """Return a resolution report and mutate ``candidate['modeling_input']`` in place.

    ``step`` is REQUIRED and has no default: the modeled reaction step must be
    declared explicitly, so an agent can never implicitly model an unspecified step.
    """
    mi = candidate["modeling_input"]
    features = _derive_features(candidate, manifest)
    family_metal = family_metal if family_metal is not None else _load_family_metal()
    coordinates = coordinates if coordinates is not None else ReactionCoordinates.load()
    atom_selectors = atom_selectors if atom_selectors is not None else load_atom_selectors()
    activation = mi.get("activation") or {}
    nucleotide = activation.get("nucleotide")
    family_ids = _candidate_family_ids(candidate)
    mech_key, mechanism, selection_basis = coordinates.select(
        family_ids=family_ids,
        activation_mode=activation.get("activation_mode"),
        override=mechanism_override,
    )
    provenance: dict[str, Any] = {}
    errors: list[str] = []

    # Resolve the declared step against the reaction coordinate (fail closed).
    step_spec = None
    if step is None:
        errors.append("modeled reaction step must be declared (no default step)")
    elif mechanism is None:
        errors.append(
            f"no reaction coordinate for activation_mode {activation.get('activation_mode')!r}"
            f" / override {mechanism_override!r}"
        )
    else:
        step_spec = coordinates.step(mechanism, step)
        if step_spec is None:
            errors.append(f"step {step} is not defined for mechanism {mech_key!r}")

    provenance["enzyme_state"] = _resolve_enzyme_state(mi, manifest, knowledge, features)
    sub_provs, sub_errors = _resolve_substrates(mi, manifest, knowledge, features)
    provenance["substrates"] = sub_provs
    errors.extend(sub_errors)
    metal_prov, metal_errors = _resolve_metal(mi, candidate, manifest, knowledge, features, fetch, family_metal)
    provenance["catalytic_metal"] = metal_prov
    errors.extend(metal_errors)

    if step_spec is not None:
        dropped = _apply_step_species(mi, step_spec, nucleotide)
        geo_prov, geo_errors = _resolve_geometry_for_step(
            mi, manifest, knowledge, features, step_spec, mech_key, nucleotide, atom_selectors
        )
        provenance["geometry"] = geo_prov
        errors.extend(geo_errors)
        errors.extend(_step_consistency_errors(mi, step_spec, nucleotide))
        step_directive = manifest.get("step") or {}
        provenance["step"] = {
            "index": step_spec.get("index"),
            "name": step_spec.get("name"),
            "mechanism": mech_key,
            "mechanism_selected_by": selection_basis,
            "candidate_family_ids": family_ids,
            "reactive_state": step_spec.get("reactive_state"),
            "chosen_because": step_directive.get("chosen_because"),
            "dropped_species": dropped,
            "coordinate_source": coordinates.source,
        }
        mi["modeled_step"] = provenance["step"]

    # Scoring policy guard + interpretation limits, grounded in the literature.
    mi["scoring_policy"] = {
        "restraints": manifest.get("scoring_policy", "unrestrained"),
        "selection": "geometry_filter_before_confidence",
        "note": (
            "No constraints block is emitted: scoring runs are unrestrained per "
            "CANDIDATE_RANKING_POLICY.md; restraints invert the activity signal."
        ),
        "literature_basis": knowledge.basis_for("pose-or-conformer-selection-criterion", features),
    }
    mi["interpretation_limits"] = knowledge.basis_for("validation-strategy", features) + knowledge.basis_for(
        "conformational-sampling-strategy", features
    )
    mi["resolution_provenance"] = provenance
    mi["knowledge_snapshot"] = {
        "source": knowledge.source,
        "schema_version": knowledge.schema_version,
        "corpus": knowledge.corpus,
    }

    # Re-derive status from the renderer's own contract.
    mi["unresolved"] = []
    mi["status"] = "ready"
    shape = execution_shape_errors(mi)
    ready = readiness_errors(mi)
    all_errors = list(dict.fromkeys(errors + shape + ready))
    if all_errors:
        mi["status"] = "needs_review"
        mi["unresolved"] = all_errors
    return {
        "accession": candidate.get("accession"),
        "status": mi["status"],
        "errors": all_errors,
        "cofactors": [f"{c.get('ccd')}x{c.get('count')}" for c in mi.get("cofactors") or []],
        "provenance": provenance,
    }


def resolve_handoff(
    handoff: dict,
    manifest: dict,
    knowledge: LiteratureKnowledge,
    *,
    step: int | None = None,
    fetch: Callable | None = None,
) -> dict:
    """Resolve every candidate; return a per-candidate report keyed by accession.

    ``step`` (CLI/caller) overrides the manifest's declared ``step.index``."""
    if handoff.get("schema_version") != "2.0":
        raise ValueError(f"unsupported handoff schema {handoff.get('schema_version')!r}; expected '2.0'")
    family_metal = _load_family_metal()
    coordinates = ReactionCoordinates.load()
    atom_selectors = load_atom_selectors()
    step_directive = manifest.get("step") or {}
    resolved_step = step if step is not None else step_directive.get("index")
    mechanism_override = step_directive.get("mechanism")
    reports = {}
    for candidate in handoff.get("candidates", []):
        reports[candidate.get("accession")] = resolve_candidate(
            candidate, manifest, knowledge,
            step=resolved_step, fetch=fetch, family_metal=family_metal,
            coordinates=coordinates, atom_selectors=atom_selectors,
            mechanism_override=mechanism_override,
        )
    handoff["resolved_by"] = {
        "tool": "modeling_reasoning.resolve_handoff",
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest.get("target_id"),
        "modeled_step": resolved_step,
        "knowledge_source": knowledge.source,
    }
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, help="schema-2.0 discovery handoff (input)")
    parser.add_argument("--manifest", required=True, help="resolution manifest (see manifests/)")
    parser.add_argument("--out", required=True, help="resolved handoff (output)")
    parser.add_argument("--report", help="per-candidate resolution report JSON")
    parser.add_argument("--knowledge", default=DEFAULT_KNOWLEDGE, help="vendored literature rules")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="reaction step to model (overrides the manifest's step.index); required if the manifest omits it",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not query UniProt; metal resolution uses manifest/default only",
    )
    args = parser.parse_args(argv)

    with open(args.handoff) as handle:
        handoff = json.load(handle)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    knowledge = LiteratureKnowledge.load(args.knowledge)
    fetch = (lambda *a, **k: None) if args.offline else None

    reports = resolve_handoff(handoff, manifest, knowledge, step=args.step, fetch=fetch)

    with open(args.out, "w") as handle:
        json.dump(handoff, handle, indent=2)
        handle.write("\n")
    if args.report:
        with open(args.report, "w") as handle:
            json.dump(reports, handle, indent=2)
            handle.write("\n")

    ready = sum(1 for r in reports.values() if r["status"] == "ready")
    print(f"resolved {len(reports)} candidates: {ready} ready, {len(reports) - ready} needs_review")
    for acc, report in reports.items():
        flag = "OK " if report["status"] == "ready" else "REV"
        print(f"  {flag} {acc:14} cofactors={','.join(report['cofactors']) or '(none)'}")
        for err in report["errors"]:
            print(f"        - {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
