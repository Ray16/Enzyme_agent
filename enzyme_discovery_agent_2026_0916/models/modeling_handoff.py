"""Validated, reaction-agnostic projection from discovery to structural modeling."""

from __future__ import annotations

from enzyme_discovery_agent.chemistry_logic import (
    derive_activation_mechanism,
    derive_modeling_cofactors,
)

SCHEMA_VERSION = "2.0"
READY_STATUSES = {"ready", "needs_review", "unresolved"}


def characterize_smiles(smiles: str) -> dict:
    """Return representation-sensitive charge and stereochemistry facts."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"valid": False, "input_smiles": smiles}
    centers = Chem.FindMolChiralCenters(
        molecule, includeUnassigned=True, includeCIP=True,
    )
    assigned = [{"atom_index": i, "cip": label} for i, label in centers if label != "?"]
    unassigned = [i for i, label in centers if label == "?"]
    if not centers:
        stereo_status = "not_applicable"
    elif unassigned:
        stereo_status = "unspecified"
    else:
        stereo_status = "specified"
    return {
        "valid": True,
        "input_smiles": smiles,
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True,
        ),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "stereochemistry": {
            "status": stereo_status,
            "assigned_centers": assigned,
            "unassigned_atom_indices": unassigned,
        },
    }


def _target_specification(target: dict) -> dict:
    reasoning = target.get("target_reasoning") or {}
    value = reasoning.get("modeling_specification")
    return value if isinstance(value, dict) else {}


def build_target_substrates(target: dict) -> tuple[list[dict], list[str]]:
    """Validate runtime role and molecular-state decisions made by the agent.

    No reaction-class rules live here. The validator checks that every source
    component was mapped and that molecular identity changes are reviewable.
    """
    source_components = [
        item.strip() for item in str(target.get("substrate_smiles") or "").split(".")
        if item.strip()
    ]
    specification = _target_specification(target)
    proposed_by_index = {
        item.get("source_component_index"): item
        for item in specification.get("components") or []
        if isinstance(item, dict) and isinstance(item.get("source_component_index"), int)
    }
    reviewed_by_index = {
        item.get("source_component_index"): item
        for item in target.get("reviewed_modeling_substrates") or []
        if isinstance(item, dict) and isinstance(item.get("source_component_index"), int)
    }
    records = []
    unresolved = []
    for index, source_smiles in enumerate(source_components, 1):
        source_facts = characterize_smiles(source_smiles)
        proposal = proposed_by_index.get(index) or {}
        review = reviewed_by_index.get(index)
        modeling_smiles = (
            (review or {}).get("modeling_smiles")
            or proposal.get("modeling_smiles")
            or source_facts.get("canonical_isomeric_smiles")
        )
        modeling_facts = (
            characterize_smiles(modeling_smiles) if modeling_smiles else {"valid": False}
        )
        changed = (
            modeling_facts.get("canonical_isomeric_smiles")
            != source_facts.get("canonical_isomeric_smiles")
        )
        record = {
            "component_index": index,
            "name": proposal.get("name"),
            "source_smiles": source_smiles,
            "source_representation": source_facts,
            "modeling_smiles": modeling_smiles,
            "modeling_representation": modeling_facts,
            "role": proposal.get("role"),
            "reactive_atom": proposal.get("reactive_atom"),
            "include_in_model": proposal.get("include_in_model", True),
            "affinity_target": proposal.get("affinity_target", False),
            "role_basis": {
                "source_type": "discovery_agent_target_reasoning",
                "source_value": proposal,
                "derivation_rule": "target_bond_change_and_participant_roles",
                "confidence": proposal.get("confidence", "unknown"),
            },
            "review_status": (
                "reviewed" if review else "required" if changed else "not_required"
            ),
        }
        if changed:
            record["state_change"] = {
                "type": "molecular_representation",
                "reason": proposal.get("state_rationale"),
            }
        if review:
            record["review_provenance"] = review.get("review_provenance")

        if not source_facts.get("valid"):
            unresolved.append(f"target component {index} source SMILES is invalid")
        if not proposal:
            unresolved.append(
                f"target component {index} has no discovery-agent modeling specification"
            )
        if not record["role"]:
            unresolved.append(f"target component {index} has no mechanistic role")
        if proposal.get("reactive") is True and not record["reactive_atom"]:
            unresolved.append(f"target component {index} has no reactive-atom mapping")
        if source_facts.get("stereochemistry", {}).get("status") == "unspecified":
            unresolved.append(f"target component {index} source stereochemistry is unspecified")
        if modeling_facts.get("stereochemistry", {}).get("status") == "unspecified":
            unresolved.append(f"target component {index} modeling stereochemistry is unspecified")
        if changed and not proposal.get("state_rationale"):
            unresolved.append(f"target component {index} state change lacks a rationale")
        if record["review_status"] == "required":
            unresolved.append(f"target component {index} modeling state requires review")
        records.append(record)

    unknown_indices = sorted(set(proposed_by_index) - set(range(1, len(source_components) + 1)))
    if unknown_indices:
        unresolved.append(f"modeling specification references unknown components: {unknown_indices}")
    if not source_components:
        unresolved.append("target substrate structures are missing")
    return records, list(dict.fromkeys(unresolved))


def _mechanism_facts(candidate: dict) -> tuple[dict, dict]:
    evidence = candidate.get("mechanism_evidence") or {}
    facts = evidence.get("facts") if isinstance(evidence, dict) else {}
    return evidence, facts if isinstance(facts, dict) else {}


def _facts_cofactors(evidence: dict, facts: dict) -> tuple[list[dict], list[str]]:
    cofactors = []
    unresolved = []
    for default_role, entries in (
        ("cofactor", facts.get("cofactors") or []),
        ("metal", facts.get("metals") or []),
    ):
        for item in entries:
            if not isinstance(item, dict):
                continue
            if not item.get("ccd"):
                unresolved.append("a mechanism cofactor lacks a structure identifier")
                continue
            if not isinstance(item.get("count"), int) or item["count"] < 1:
                unresolved.append("a mechanism cofactor lacks explicit stoichiometry")
                continue
            cofactors.append({
                **item,
                "role": item.get("role") or default_role,
                "basis": {
                    "source_type": "mechanism_literature",
                    "source_record": evidence.get("paper") or {},
                    "source_value": item,
                    "confidence": facts.get("confidence"),
                },
                "confidence": facts.get("confidence"),
            })
    return cofactors, unresolved


def _merge_cofactors(*groups: list[dict]) -> tuple[list[dict], list[str]]:
    """Merge independent evidence without silently replacing or duplicating entities."""
    merged = []
    by_identity = {}
    unresolved = []
    for group in groups:
        for item in group:
            identity = (item.get("ccd"), item.get("role"))
            existing = by_identity.get(identity)
            if existing is None:
                copied = dict(item)
                by_identity[identity] = copied
                merged.append(copied)
                continue
            if existing.get("count") != item.get("count"):
                unresolved.append(
                    f"conflicting stoichiometry for {item.get('ccd') or 'a cofactor'}"
                )
    return merged, unresolved


def _candidate_geometry(
    candidate: dict,
    target_specification: dict,
    activation: dict | None,
    evidence: dict,
    facts: dict,
) -> tuple[dict | None, list[str]]:
    """Project geometry without fixed metrics, family names, or target-class rules."""
    explicit = candidate.get("modeling_specification") or {}
    if isinstance(explicit.get("geometry"), dict):
        return explicit["geometry"], []

    if activation and activation.get("status") == "assigned":
        target_geometry = target_specification.get("geometry") or {}
        attacking_participant = (
            target_geometry.get("nucleophile") or target_geometry.get("atom_a")
        )
        if not attacking_participant:
            return None, ["target reasoning does not identify the attacking participant"]
        return {
            "kind": "candidate_activation",
            "attacking_participant": attacking_participant,
            "activated_center": {
                "source": "cofactor",
                "identity": activation["nucleotide"],
                "atom": activation["reactive_atom"],
            },
            "leaving_group": {
                "source": "cofactor",
                "identity": activation["nucleotide"],
                "atom": activation["leaving_group_atom"],
            },
            "basis": activation["basis"],
        }, []

    if facts.get("is_clean_inline_substitution") is True:
        participants = {
            "nucleophile": facts.get("nucleophile") or {},
            "electrophile": facts.get("electrophile") or {},
            "leaving_group": facts.get("leaving_group") or {},
        }
        if all(value.get("atom") for value in participants.values()):
            return {
                "kind": "inline_substitution",
                **participants,
                "basis": {
                    "source_type": "mechanism_literature",
                    "source_record": evidence.get("paper") or {},
                    "confidence": facts.get("confidence"),
                },
            }, []
        return None, ["mechanism facts do not identify all required geometry atoms"]

    target_geometry = target_specification.get("geometry")
    if isinstance(target_geometry, dict) and target_geometry.get("kind") not in (None, "none"):
        return None, ["target geometry is proposed but not grounded for this candidate"]
    return None, ["no geometry observable is defined for this mechanism"]


def build_modeling_input(candidate: dict, target: dict) -> dict:
    """Build a conservative, reaction-agnostic candidate modeling specification."""
    reactions = candidate.get("catalytic_activities") or []
    evidence, facts = _mechanism_facts(candidate)
    activation = derive_activation_mechanism(reactions)
    ntp_projection = derive_modeling_cofactors(
        reactions,
        candidate.get("required_cofactors") or [],
        candidate.get("cofactor_notes") or [],
    )
    fact_cofactors, fact_unresolved = _facts_cofactors(evidence, facts)
    cofactors, merge_unresolved = _merge_cofactors(
        ntp_projection["cofactors"], fact_cofactors,
    )
    cofactor_unresolved = (
        ntp_projection["unresolved"] + fact_unresolved + merge_unresolved
    )

    substrates, unresolved = build_target_substrates(target)
    unresolved.extend(cofactor_unresolved)
    if activation and activation.get("status") == "conflicting":
        unresolved.append("candidate reaction has conflicting activation signatures")
    elif activation and activation.get("confidence") != "high":
        unresolved.append("candidate activation lacks a corroborating product signature")

    target_specification = _target_specification(target)
    geometry, geometry_unresolved = _candidate_geometry(
        candidate, target_specification, activation, evidence, facts,
    )
    unresolved.extend(target_specification.get("uncertainties") or [])
    enzyme_chains = (
        (candidate.get("modeling_specification") or {}).get("enzyme_chains")
    )
    enzyme_state = facts.get("enzyme_state") or {}
    if enzyme_chains is None:
        if enzyme_state.get("state") == "intact":
            enzyme_chains = [{"source": "full_sequence"}]
        elif enzyme_state.get("state") == "processed":
            mature_chains = enzyme_state.get("mature_chains") or []
            if mature_chains and all(
                isinstance(chain.get("start"), int)
                and isinstance(chain.get("end"), int)
                and chain["start"] <= chain["end"]
                for chain in mature_chains
            ):
                enzyme_chains = [
                    {
                        "source": "sequence_range",
                        "start": chain["start"],
                        "end": chain["end"],
                    }
                    for chain in mature_chains
                ]
            else:
                unresolved.append(
                    "processed enzyme lacks source-supported mature-chain boundaries"
                )
        else:
            unresolved.append("catalytically relevant enzyme state is unresolved")

    status = "ready" if not unresolved else "needs_review"
    if not substrates or not enzyme_chains:
        status = "unresolved"
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "method": (
            (candidate.get("modeling_specification") or {}).get("method")
            or target_specification.get("method")
            or "cofold"
        ),
        "state": (
            (candidate.get("modeling_specification") or {}).get("state")
            or target_specification.get("state")
            or "unspecified"
        ),
        "enzyme_state": {
            "state": enzyme_state.get("state", "specified"),
            "chains": enzyme_chains,
            "basis": {
                "source_type": (
                    "candidate_modeling_specification"
                    if (candidate.get("modeling_specification") or {}).get("enzyme_chains")
                    else "mechanism_literature"
                ),
                "source_record": evidence.get("paper") or {},
            },
        },
        "target_substrates": substrates,
        "activation": activation,
        "cofactors": cofactors,
        "geometry": geometry,
        "geometry_status": "ready" if geometry else "unresolved",
        "unresolved": list(dict.fromkeys(unresolved)),
        "warnings": list(dict.fromkeys(geometry_unresolved)),
    }
    errors = validate_modeling_input(result)
    if errors:
        result["status"] = "unresolved"
        result["unresolved"].extend(
            error for error in errors if error not in result["unresolved"]
        )
    return result


def validate_modeling_input(modeling_input: dict) -> list[str]:
    """Return execution-blocking schema or chemistry errors."""
    errors = []
    if modeling_input.get("status") not in READY_STATUSES:
        errors.append("invalid modeling_input.status")
    for substrate in modeling_input.get("target_substrates") or []:
        if not substrate.get("modeling_representation", {}).get("valid"):
            errors.append("invalid target modeling SMILES")
        if substrate.get("modeling_representation", {}).get(
            "stereochemistry", {},
        ).get("status") == "unspecified":
            errors.append("modeling substrate has unspecified stereochemistry")
    geometry = modeling_input.get("geometry") or {}
    if geometry and not any(
        geometry.get(key)
        for key in ("activated_center", "electrophile", "atom_b", "observable")
    ):
        errors.append("modeling geometry lacks a second reactive participant")
    return list(dict.fromkeys(errors))


def is_execution_ready(modeling_input: dict) -> tuple[bool, list[str]]:
    errors = validate_modeling_input(modeling_input)
    if modeling_input.get("status") != "ready":
        errors.append(f"modeling input status is {modeling_input.get('status', 'missing')}")
    errors.extend(modeling_input.get("unresolved") or [])
    for substrate in modeling_input.get("target_substrates") or []:
        if substrate.get("review_status") == "required":
            errors.append("a substrate representation change has not been reviewed")
    return not errors, list(dict.fromkeys(errors))
