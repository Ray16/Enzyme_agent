"""The shared, derived chemistry contract for a discovery run.

One structured account of what chemistry the target requires, derived ONCE at
run start and consumed by every downstream stage (discovery, screening,
candidate evaluation, ranking), so no stage independently invents a different
account of the required chemistry.

Mechanical facts are extracted deterministically (status ``derived_from_target``);
genuinely inferential claims are reasoned by the model and recorded as
hypotheses with explicit uncertainty (status ``inferred``).  Every claim carries
a basis and an uncertainty so a reader can tell an established fact from an
inference.

Firewall: this module derives the contract from the user-supplied target only
(the reaction the user described and the substrates the user named).  It encodes
no answer families, substrates, product names, or enzyme/gene names.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")

SCHEMA_VERSION = "1.0"
STATUSES = frozenset({"derived_from_target", "inferred", "assumed"})
UNCERTAINTIES = frozenset({"low", "medium", "high"})
DIRECTIONS = frozenset({"cleavage", "formation", "transfer", "redox", "other"})
TOPOLOGIES = frozenset({
    "free_small_molecule", "carrier_protein_tethered", "polymer",
    "macromolecule", "mixed",
})

_LLM_SYSTEM = (
    "You are a mechanistic enzymologist deriving the CHEMICAL constraints a "
    "reaction places on any catalyst, from the reaction description and the "
    "substrate structures alone. You are given deterministically extracted facts "
    "(bond change, reaction direction, substrate functional groups, substrate "
    "topology) as a cross-check — trust the reaction description over them if they "
    "disagree. Reason ONLY about chemistry — never name a specific enzyme, gene, "
    "family, or organism, and never assume the answer. Return STRICT JSON:\n"
    "{\n"
    '  "reaction_direction": "cleavage|formation|transfer|redox|other — the direction '
    'the TARGET reaction runs (e.g. hydrolysis/degradation/depolymerization = cleavage; '
    'ligation/synthesis/condensation = formation)",\n'
    '  "bond_types": ["the bond type(s) made, broken or transformed in the supplied reaction"],\n'
    '  "substrate_topology": "free_small_molecule|carrier_protein_tethered|polymer|'
    'macromolecule|mixed — the physical form of the target substrate INPUT as described '
    '(if the input is a discrete small-molecule fragment used to stand in for a polymer, '
    'report what the described structure actually is)",\n'
    '  "activation_problem": "one sentence on the thermodynamic/activation barrier",\n'
    '  "activation_categories": ["generic categories of activated intermediate or '
    'coupled favorable process that could solve it"],\n'
    '  "cofactor_requirements": ["cofactors/leaving groups chemically implied, if any"],\n'
    '  "disqualifiers": ["observations about a candidate that would rule it out"],\n'
    '  "modeling_specification": {\n'
    '    "state": "<chemical state represented by the proposed complex>",\n'
    '    "components": [\n'
    '      {"source_component_index": 1, "name": "<target molecule>", '
    '"role": "<mechanistic role>", '
    '"modeling_smiles": "<isomeric SMILES in the proposed modeled state>", '
    '"state_rationale": "<basis for charge/protonation state>", '
    '"reactive_atom": "<unambiguous atom description or null>", '
    '"include_in_model": true, "affinity_target": true|false}\n'
    "    ],\n"
    '    "geometry": {"kind": "<distance|distance_angle|other|none>", '
    '"atom_a": "<participant/atom description or null>", '
    '"atom_b": "<participant/atom description or null>", '
    '"atom_c": "<participant/atom description or null>", '
    '"rationale": "<chemical basis or null>"},\n'
    '    "uncertainties": ["<choice that requires evidence or review>"]\n'
    "  },\n"
    '  "polymer_target": true|false,  // does the reaction BUILD or BREAK a repeating '
    'polymer/oligomer chain, even if the named inputs are monomers (i.e. the target '
    'reaction synthesizes or degrades a repeating chain)?\n'
    '  "repeat_unit": "if polymer_target, a short phrase for the repeating unit (else null)",\n'
    '  "oligomer_smiles": "if polymer_target, a VALID SMILES for a representative oligomer '
    'of at least four residues (a few repeat units) that a structural model should use to '
    'stand in for the chain, built from the named monomers/repeat unit; else null",\n'
    '  "uncertainty": "low|medium|high"\n'
    "}\n"
    "For modeling_specification, preserve every stereocenter stated by a substrate "
    "name or source structure and never choose an unspecified stereoisomer. Keep "
    "source and proposed modeled molecular states separate. Propose a different "
    "protonation or charge state only when the modeled chemical step requires it, "
    "explain why, and list the change in uncertainties so it is reviewed. A reacting "
    "nucleophile must be represented in a nucleophilic state. Assign roles from the "
    "target bond change, not component order. Atom descriptions must be specific "
    "enough to map to a structure; use null when they are not. "
    "Keep every list short and general. If a field is not chemically implied, use "
    "an empty list (or null for the scalar fields). The oligomer, when given, must be a "
    "chemically valid extension of the target chemistry, not a copy of a single monomer."
)


def _claim(claim: str, basis: str, implication: str, status: str,
           uncertainty: str) -> dict:
    if status not in STATUSES:
        raise ValueError(f"invalid claim status: {status}")
    if uncertainty not in UNCERTAINTIES:
        raise ValueError(f"invalid claim uncertainty: {uncertainty}")
    return {
        "claim": claim,
        "basis": basis,
        "implication": implication,
        "status": status,
        "uncertainty": uncertainty,
    }


def infer_substrate_topology(substrate_smiles: str | None,
                             substrate_profile: dict | None) -> str | None:
    """Normalize the physical form of the target INPUTS (never enzyme capability).

    Shared, pure form of the logic previously inlined in
    ``family_reactions._target_topology_for_run`` so the contract and that reader
    agree.  Explicit profile topology wins; otherwise infer from descriptive
    fields; a dot-separated SMILES mixture is a set of free input molecules.
    """
    substrate = substrate_profile or {}
    explicit = str(substrate.get("topology") or substrate.get("class") or "").strip()
    if explicit in TOPOLOGIES:
        return explicit
    description = " ".join(str(substrate.get(k) or "") for k in (
        "physical_state", "access_constraint",
    )).casefold()
    if "carrier" in description and ("tether" in description or "bound" in description):
        return "carrier_protein_tethered"
    if "polymer" in description and "monomer" not in description:
        return "polymer"
    if "free" in description or "soluble monomer" in description:
        return "free_small_molecule"
    if substrate_smiles:
        return "free_small_molecule"
    return None


def _reactive_groups(substrate_smiles: str | None) -> list[str]:
    """Union of functional groups across the target substrate structures.

    Deterministic (RDKit/SMARTS via tools.substrate).  A dot-separated SMILES is
    a mixture of discrete inputs, so each component is analyzed and the groups
    unioned.  Returns [] when structures are absent or unparseable.
    """
    if not substrate_smiles:
        return []
    try:
        from enzyme_discovery_agent.tools.substrate import compute_substrate_properties
    except Exception as exc:  # noqa: BLE001 - RDKit missing, degrade to none
        logger.debug("reactive-group extraction unavailable: %s", exc)
        return []
    groups: list[str] = []
    for component in str(substrate_smiles).split("."):
        component = component.strip()
        if not component:
            continue
        try:
            props = compute_substrate_properties(component)
        except Exception:  # noqa: BLE001 - unparseable component, skip
            continue
        for group in props.functional_groups:
            if group not in groups:
                groups.append(group)
    return groups


# Backbone-linkage SMARTS for the common target bond types, used only to sanity-
# check that an LLM-proposed oligomer really is a multi-residue chain of the right
# kind. Generic chemistry (no target/answer is encoded here).
_BOND_SMARTS = {
    "amide": "[NX3][CX3](=O)",
    "peptide": "[NX3][CX3](=O)",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "thioester": "[#6][CX3](=O)[SX2][#6]",
    "carbamate": "[NX3][CX3](=O)[OX2][#6]",
}


def _backbone_bond_count(mol, bond_types) -> tuple[int, bool]:
    """(max linkages of any recognized target bond type, whether any was recognized)."""
    from rdkit import Chem  # local import: RDKit may be unavailable
    best, recognized = 0, False
    for bt in bond_types or []:
        key = str(bt).strip().casefold()
        smarts = next((s for k, s in _BOND_SMARTS.items() if k in key), None)
        if not smarts:
            continue
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        recognized = True
        best = max(best, len(mol.GetSubstructMatches(patt)))
    return best, recognized


def _largest_monomer_mw(substrate_smiles: str | None) -> float:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mw = 0.0
    for comp in str(substrate_smiles or "").split("."):
        comp = comp.strip()
        if not comp:
            continue
        m = Chem.MolFromSmiles(comp)
        if m is not None:
            mw = max(mw, Descriptors.MolWt(m))
    return mw


def validate_polymer_oligomer(smiles: str | None, bond_types, substrate_smiles: str | None,
                              min_residues: int = 4) -> dict | None:
    """Validate an LLM-proposed representative oligomer for a polymer target.

    Returns ``{smiles, n_backbone_bonds, ...}`` for a credible oligomer, or None
    when it is missing, unparseable, or not clearly a multi-residue chain.  A
    recognized backbone bond type must appear at least ``min_residues - 1`` times
    (four residues -> three linkages); when the bond type is not one we can match
    by SMARTS, the oligomer must at least be substantially larger than the largest
    named monomer.  Answer-agnostic: only the target's own chemistry is used.
    """
    smiles = str(smiles or "").strip()
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception as exc:  # noqa: BLE001 - RDKit missing, cannot validate
        logger.debug("oligomer validation unavailable: %s", exc)
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_bonds, recognized = _backbone_bond_count(mol, bond_types)
    if recognized:
        if n_bonds < max(1, min_residues - 1):
            return None
    else:
        mono = _largest_monomer_mw(substrate_smiles)
        if mono and Descriptors.MolWt(mol) < 2.0 * mono:
            return None
    return {
        "smiles": Chem.MolToSmiles(mol),
        "n_backbone_bonds": n_bonds,
        "n_residues_min": min_residues,
        "basis": "llm_derived",
    }


def _deterministic_layer(query: str, substrate_smiles: str | None,
                         target_profile: dict | None) -> tuple[dict, list[dict]]:
    """Extract the mechanical, high-confidence facts and their claims."""
    profile = target_profile or {}
    reaction = profile.get("reaction") or {}
    substrate = profile.get("substrate") or {}
    target_text = " ".join(filter(None, [
        str(reaction.get("class", "")),
        str(reaction.get("direction", "")),
        str(query or ""),
    ]))
    bond_info = extract_target_bond_type(target_text)
    direction = bond_info.get("direction")
    bond_types = list(bond_info.get("bond_types") or [])
    topology = infer_substrate_topology(substrate_smiles, substrate)
    reactive_groups = _reactive_groups(substrate_smiles)

    claims: list[dict] = []
    if direction:
        # An explicit profile direction is a stated requirement (low uncertainty);
        # one parsed from free text is a confident read but still an extraction.
        explicit_dir = str(reaction.get("direction", "")).strip().casefold() == direction
        claims.append(_claim(
            f"The required reaction direction is {direction}.",
            "stated transformation" if explicit_dir else "reaction verb in the target description",
            "a catalyst whose native reaction runs the opposite direction is disqualified",
            "derived_from_target", "low" if explicit_dir else "medium",
        ))
    for bond in bond_types:
        claims.append(_claim(
            f"The reaction changes a {bond} bond.",
            "bond-type keyword in the target description",
            "the catalyst must act on this bond class",
            "derived_from_target", "medium",
        ))
    if topology:
        explicit_top = str(substrate.get("topology") or substrate.get("class") or "").strip() in TOPOLOGIES
        claims.append(_claim(
            f"The target inputs have {topology} topology.",
            "stated substrate form" if explicit_top else "structure of the named substrates",
            "a catalyst requiring a different substrate topology may not accept these inputs",
            "derived_from_target", "low" if explicit_top else "medium",
        ))
    if reactive_groups:
        claims.append(_claim(
            "The target substrates carry these reactive groups: "
            + ", ".join(reactive_groups) + ".",
            "functional groups detected in the substrate structures",
            "the catalytic chemistry must engage one or more of these groups",
            "derived_from_target", "low",
        ))

    fields = {
        "reaction_direction": direction,
        "bond_types": bond_types,
        "substrate_topology": topology,
        "reactive_groups": reactive_groups,
    }
    return fields, claims


def _llm_layer(query: str, substrate_smiles: str | None, fields: dict,
               llm_client, on_response=None) -> tuple[dict, list[dict]]:
    """Reason the inferential claims from the deterministic facts.

    Degrades to empty (deterministic-only contract) when no client is available
    or the call fails/does not parse.  ``on_response`` (if given) is invoked with
    the raw LLM response as soon as it is available, so a caller can account for
    the tokens this derivation spent even when parsing later fails.
    """
    empty = ({"activation_problem": "", "activation_categories": [],
              "cofactor_requirements": [], "disqualifiers": [],
              "reaction_direction": None, "bond_types": [],
              "substrate_topology": None,
              "polymer_target": False, "modeling_oligomer": None,
              "modeling_specification": None}, [])
    if llm_client is None:
        return empty
    facts = {
        "reaction_direction": fields.get("reaction_direction"),
        "bond_types": fields.get("bond_types"),
        "substrate_topology": fields.get("substrate_topology"),
        "reactive_groups": fields.get("reactive_groups"),
    }
    user = (
        f"Reaction description: {query}\n"
        f"Substrate SMILES: {substrate_smiles or '(not provided)'}\n"
        f"Deterministically extracted facts: {json.dumps(facts)}"
    )
    try:
        response = llm_client.call(
            messages=[{"role": "system", "content": _LLM_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=config.LLM_TEMPERATURE,
            timeout=max(config.LLM_TIMEOUT, 60),
            max_tokens=2048,
        )
        if response is not None and on_response is not None:
            try:
                on_response(response)
            except Exception:  # noqa: BLE001 - accounting must never break derivation
                pass
        content = (response.choices[0].message.content or "") if response else ""
        obj = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    except Exception as exc:  # noqa: BLE001 - degrade to deterministic-only
        logger.debug("target-reasoning inference failed: %s", exc)
        return empty

    def _terms(value) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out

    problem = str(obj.get("activation_problem") or "").strip()
    activation = _terms(obj.get("activation_categories"))
    cofactors = _terms(obj.get("cofactor_requirements"))
    disqualifiers = _terms(obj.get("disqualifiers"))
    uncertainty = str(obj.get("uncertainty") or "").strip().casefold()
    if uncertainty not in UNCERTAINTIES:
        uncertainty = "medium"

    claims: list[dict] = []
    if problem:
        claims.append(_claim(
            problem, "mechanistic reasoning from the extracted facts",
            "the catalyst must resolve this barrier", "inferred", uncertainty,
        ))
    if activation:
        claims.append(_claim(
            "Plausible activation strategies: " + ", ".join(activation) + ".",
            "mechanistic reasoning from the extracted facts",
            "a genuine candidate should realize one of these strategies",
            "inferred", uncertainty,
        ))
    for cofactor in cofactors:
        claims.append(_claim(
            f"A cofactor or leaving group is chemically implied: {cofactor}.",
            "mechanistic reasoning from the extracted facts",
            "the catalyst likely requires this cofactor/leaving group",
            "inferred", uncertainty,
        ))
    for disqualifier in disqualifiers:
        claims.append(_claim(
            f"Disqualifying observation: {disqualifier}.",
            "mechanistic reasoning from the extracted facts",
            "a candidate showing this is ruled out",
            "inferred", uncertainty,
        ))

    # Semantic determination of direction/bond/topology by the LLM (validated
    # against the known vocabularies; unrecognized -> None so the caller falls
    # back to the deterministic parser).
    llm_direction = str(obj.get("reaction_direction") or "").strip().casefold()
    llm_direction = llm_direction if llm_direction in DIRECTIONS else None
    llm_bonds = _terms(obj.get("bond_types"))
    llm_topology = str(obj.get("substrate_topology") or "").strip()
    llm_topology = llm_topology if llm_topology in TOPOLOGIES else None

    # Polymer targets: when the reaction builds or breaks a repeating chain (even
    # from monomeric inputs), have the model reason out a representative oligomer
    # (>= a tetramer of residues) for structural modeling to use in place of a
    # single monomer or dimer. The proposal is RDKit-validated before it is kept.
    polymer_target = bool(obj.get("polymer_target"))
    modeling_oligomer = None
    if polymer_target:
        modeling_oligomer = validate_polymer_oligomer(
            obj.get("oligomer_smiles"),
            llm_bonds or fields.get("bond_types"),
            substrate_smiles,
        )
        if modeling_oligomer is not None:
            repeat_unit = str(obj.get("repeat_unit") or "").strip() or None
            modeling_oligomer["repeat_unit"] = repeat_unit
            n_res = modeling_oligomer["n_backbone_bonds"] + 1
            claims.append(_claim(
                f"The target is a repeating chain ({repeat_unit or 'polymer'}); a "
                f"representative oligomer of about {n_res} residues should stand in for "
                f"the chain in structural modeling, not a single monomer or dimer.",
                "mechanistic reasoning about the target's polymeric context",
                "structural modeling should use the derived oligomer as the substrate",
                "inferred", uncertainty,
            ))
        elif obj.get("oligomer_smiles"):
            logger.warning("target-reasoning: LLM oligomer proposal rejected as not a "
                           "credible multi-residue chain")

    modeling_specification = obj.get("modeling_specification")
    if not isinstance(modeling_specification, dict):
        modeling_specification = None

    extra = {
        "activation_problem": problem,
        "activation_categories": activation,
        "cofactor_requirements": cofactors,
        "disqualifiers": disqualifiers,
        "reaction_direction": llm_direction,
        "bond_types": llm_bonds,
        "substrate_topology": llm_topology,
        "polymer_target": polymer_target,
        "modeling_oligomer": modeling_oligomer,
        "modeling_specification": modeling_specification,
    }
    return extra, claims


def derive_target_reasoning(
    query: str,
    substrate_smiles: str | None = None,
    substrate_name: str | None = None,
    llm_client=None,
    target_profile: dict | None = None,
    on_response=None,
) -> dict:
    """Derive the full chemistry contract for a run (deterministic + inferred).

    ``on_response`` (optional) is called with the raw LLM response from the
    inferential layer, letting a caller record the tokens this derivation spent.
    """
    fields, det_claims = _deterministic_layer(query, substrate_smiles, target_profile)
    extra, llm_claims = _llm_layer(
        query, substrate_smiles, fields, llm_client, on_response=on_response)

    # The LLM's semantic reading is authoritative for direction and topology
    # (string-matching is brittle — it silently missed "hydrolyze/hydrolases");
    # the deterministic parser is the fallback when the LLM is unavailable or
    # returned an unrecognized value. Bond types stay deterministic-primary (already
    # normalized/reliable), with the LLM only filling in when the parser found none.
    def _prefer(llm_value, det_value):
        return llm_value if llm_value else det_value

    direction = _prefer(extra.get("reaction_direction"), fields["reaction_direction"])
    topology = _prefer(extra.get("substrate_topology"), fields["substrate_topology"])
    bond_types = fields["bond_types"] or extra.get("bond_types") or []
    for name, llm_v, det_v in (
        ("reaction_direction", extra.get("reaction_direction"), fields["reaction_direction"]),
        ("substrate_topology", extra.get("substrate_topology"), fields["substrate_topology"]),
    ):
        if llm_v and det_v and llm_v != det_v:
            logger.warning("target-reasoning %s disagreement: llm=%s deterministic=%s "
                           "(using llm)", name, llm_v, det_v)

    contract = {
        "schema_version": SCHEMA_VERSION,
        "derived_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "substrate_smiles": substrate_smiles,
        "substrate_name": substrate_name,
        "claims": det_claims + llm_claims,
        # Machine-readable fields the guards consume: LLM-authoritative for
        # direction/topology, deterministic fallback (see above).
        "reaction_direction": direction,
        "bond_types": bond_types,
        "substrate_topology": topology,
        "reactive_groups": fields["reactive_groups"],
        "activation_problem": extra["activation_problem"],
        "activation_categories": extra["activation_categories"],
        "cofactor_requirements": extra["cofactor_requirements"],
        "disqualifiers": extra["disqualifiers"],
        # Polymer-aware modeling substrate: a representative oligomer the agent
        # reasoned out for a polymer target (None for non-polymer targets).
        "polymer_target": bool(extra.get("polymer_target")),
        "modeling_oligomer": extra.get("modeling_oligomer"),
        "modeling_specification": extra.get("modeling_specification"),
    }
    return normalize_target_reasoning(contract)


def normalize_target_reasoning(contract: dict | None) -> dict:
    """Validate and normalize a target-reasoning contract."""
    if contract is None:
        raise ValueError("target reasoning contract is required.")
    if not isinstance(contract, dict):
        raise ValueError("target reasoning must be a JSON object.")
    if contract.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Unsupported target-reasoning schema; expected version 1.0.")
    result = dict(contract)
    result["schema_version"] = SCHEMA_VERSION

    claims = result.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims must be a list.")
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("each claim must be an object.")
        for key in ("claim", "basis", "implication", "status", "uncertainty"):
            if key not in claim:
                raise ValueError(f"claim is missing required field '{key}'.")
        if claim["status"] not in STATUSES:
            raise ValueError(f"invalid claim status: {claim['status']}")
        if claim["uncertainty"] not in UNCERTAINTIES:
            raise ValueError(f"invalid claim uncertainty: {claim['uncertainty']}")

    result.setdefault("reaction_direction", None)
    result["bond_types"] = list(result.get("bond_types") or [])
    result.setdefault("substrate_topology", None)
    result["reactive_groups"] = list(result.get("reactive_groups") or [])
    result["activation_categories"] = list(result.get("activation_categories") or [])
    result["cofactor_requirements"] = list(result.get("cofactor_requirements") or [])
    result["disqualifiers"] = list(result.get("disqualifiers") or [])
    result.setdefault("activation_problem", "")
    result.setdefault("polymer_target", False)
    result.setdefault("modeling_oligomer", None)
    modeling = result.get("modeling_specification")
    if modeling is not None and not isinstance(modeling, dict):
        raise ValueError("modeling_specification must be an object or null.")
    result.setdefault("modeling_specification", None)
    topology = result["substrate_topology"]
    if topology is not None and topology not in TOPOLOGIES:
        raise ValueError(f"invalid substrate_topology: {topology}")
    return result


def load_target_reasoning(run_dir: str | Path) -> dict | None:
    """Load a run's persisted chemistry contract, or None if absent/unreadable."""
    path = run_file(Path(run_dir), 'target_reasoning.json')
    try:
        return normalize_target_reasoning(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def target_claims(run_dir: str | Path) -> tuple[str | None, set[str], str | None]:
    """The authoritative (direction, bond_types, topology) for a run.

    The single reader every downstream guard should call.  Returns
    ``(None, set(), None)`` when no contract has been persisted so a legacy
    caller can fall back to its own extraction.
    """
    contract = load_target_reasoning(run_dir)
    if contract is None:
        return None, set(), None
    return (
        contract.get("reaction_direction"),
        set(contract.get("bond_types") or []),
        contract.get("substrate_topology"),
    )
