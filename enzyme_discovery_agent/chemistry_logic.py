"""Reaction chemistry utilities: classification, bond-type matching, substrate comparison.

Pure functions with no imports from models/ to avoid circular dependencies.
Extracted from the original tools/reasoning.py.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

from enzyme_discovery_agent import config
from enzyme_discovery_agent.static.ec_bond_map import EC_BOND_TYPE_MAP
from enzyme_discovery_agent.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")

# --------------------------------------------------------------------------
# ChEBI SMILES resolution (module-level cache)
# --------------------------------------------------------------------------
_chebi_smiles_cache: dict[str, str | None] = {}
_chebi_name_cache: dict[str, str | None] = {}


def resolve_chebi_smiles(chebi_id: str) -> str | None:
    """Resolve a ChEBI ID to a SMILES string via the EBI OLS4 API."""
    if not chebi_id.startswith("CHEBI:"):
        chebi_id = f"CHEBI:{chebi_id}"

    if chebi_id in _chebi_smiles_cache:
        return _chebi_smiles_cache[chebi_id]

    numeric_id = chebi_id.replace("CHEBI:", "")
    iri = urllib.parse.quote_plus(f"http://purl.obolibrary.org/obo/CHEBI_{numeric_id}")
    url = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms?iri={iri}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        data = json.loads(request_with_retry(req, timeout=15).decode())
        terms = data.get("_embedded", {}).get("terms", [])
        for term in terms:
            annotation = term.get("annotation", {})
            smiles_list = annotation.get("smiles_string", [])
            if smiles_list:
                smiles = smiles_list[0].strip()
                _chebi_smiles_cache[chebi_id] = smiles
                name = term.get("label")
                if name:
                    _chebi_name_cache[chebi_id] = name
                return smiles
            name = term.get("label")
            if name:
                _chebi_name_cache[chebi_id] = name
    except Exception as e:
        logger.debug("ChEBI SMILES lookup failed for %s: %s", chebi_id, e)

    logger.warning("ChEBI SMILES not found for %s, caching as unavailable", chebi_id)
    _chebi_smiles_cache[chebi_id] = None
    return None


def get_chebi_name(chebi_id: str) -> str | None:
    """Return cached ChEBI compound name, or None if not yet resolved."""
    if not chebi_id.startswith("CHEBI:"):
        chebi_id = f"CHEBI:{chebi_id}"
    return _chebi_name_cache.get(chebi_id)


_name_smiles_cache: dict[str, str | None] = {}


def _pubchem_name_to_smiles(name: str) -> str | None:
    """Resolve one chemical name to a canonical SMILES via PubChem PUG REST."""
    key = name.strip().lower()
    if not key:
        return None
    if key in _name_smiles_cache:
        return _name_smiles_cache[key]
    smiles = None
    # Try the modern property token first, then the legacy one: PubChem renamed
    # CanonicalSMILES -> SMILES/ConnectivitySMILES across API versions, and which
    # one a deployment serves is not stable.
    for prop in ("SMILES", "CanonicalSMILES", "IsomericSMILES"):
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{urllib.parse.quote(name.strip())}/property/{prop}/JSON")
        try:
            raw = request_with_retry(
                urllib.request.Request(url, headers={"Accept": "application/json"}),
                timeout=20,
            )
            props = json.loads(raw.decode()).get("PropertyTable", {}).get("Properties", [])
            for entry in props:
                val = entry.get(prop) or entry.get("SMILES") or entry.get("CanonicalSMILES")
                if val:
                    smiles = val.strip()
                    break
        except Exception as e:  # noqa: BLE001 - try the next property token
            logger.debug("PubChem name->SMILES failed for %s (%s): %s", name, prop, e)
        if smiles:
            break
    _name_smiles_cache[key] = smiles
    return smiles


_SUBSTRATE_REASON_SYSTEM = (
    "At the start of an enzyme-discovery task you reason about the target reaction "
    "and its substrates. From the query, identify the CHEMICAL SUBSTRATES (the "
    "reaction inputs / reactants the enzyme must act on) and give each one's "
    "SMILES, reasoning as a chemist from the compound names in the query. "
    "Return ONLY JSON: {\"substrates\":[{\"name\":\"..\",\"smiles\":\"..\"}]}. "
    "Substrates only — not enzymes, not products, not solvents or cofactors. Give "
    "a valid canonical SMILES for each. If the query names no specific substrate, "
    'return {"substrates":[]}.'
)


def reason_target_substrate_smiles(query: str, llm_client) -> str | None:
    """Reason the target reaction's substrate SMILES from a free-text query.

    The agent already reasons about the reaction and substrates at the initial
    stage, so the SMILES are REASONED by the model (it knows common reagents),
    then validated with RDKit; an invalid/blank one falls back to a PubChem
    name lookup.  Answer-agnostic: these are the reaction INPUTS the user named
    (task setup), never any enzyme.  Returns a '.'-joined SMILES string, or None.
    """
    if not query or llm_client is None:
        return None
    try:
        response = llm_client.call(
            messages=[{"role": "system", "content": _SUBSTRATE_REASON_SYSTEM},
                      {"role": "user", "content": query}],
            temperature=config.LLM_TEMPERATURE,
            timeout=max(config.LLM_TIMEOUT, 60),
        )
        content = response.choices[0].message.content or ""
        obj = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    except Exception as exc:  # noqa: BLE001 - degrade to the raw-text path
        logger.debug("substrate reasoning failed: %s", exc)
        return None

    try:
        from rdkit import Chem
    except Exception:  # noqa: BLE001
        Chem = None
    out = []
    for s in obj.get("substrates") or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip()
        smi = str(s.get("smiles", "")).strip()
        valid = None
        if smi and Chem is not None and Chem.MolFromSmiles(smi) is not None:
            valid = Chem.CanonSmiles(smi)          # reasoned SMILES, validated
        elif name:
            valid = _pubchem_name_to_smiles(name)  # fallback: resolve the name
        if valid:
            out.append(valid)
    return ".".join(out) if out else None


def resolve_substrate_names_to_smiles(text: str | None) -> str | None:
    """Resolve a free-text substrate description to a dot-joined SMILES string.

    The target substrates come straight from the user's query (they are the
    reaction inputs, not any answer enzyme), so resolving them to structure is
    task setup, not leakage.  Splits a phrase like "A and B (context note)" into
    individual reagent names, drops parenthetical qualifiers, and resolves each
    via PubChem; returns the components joined by '.' (RDKit's mixture form) so
    substrate-accommodation comparison has real structures to score against.
    Returns None if nothing resolves, leaving the axis inactive rather than
    guessing.
    """
    if not text:
        return None
    core = re.sub(r"\(.*?\)", " ", text)          # drop parenthetical context notes
    parts = re.split(r"\band\b|[,;+/]", core, flags=re.IGNORECASE)
    names = [p.strip(" .") for p in parts if p.strip(" .")]
    resolved: list[str] = []
    for name in names:
        # Skip obvious non-reagent words that survive splitting.
        if len(name) < 3 or name.lower() in {"monomer", "monomers", "substrate", "substrates"}:
            continue
        smi = _pubchem_name_to_smiles(name)
        if smi:
            resolved.append(smi)
    return ".".join(resolved) if resolved else None


# --------------------------------------------------------------------------
# Reaction chemistry classification
# --------------------------------------------------------------------------

def classify_reaction_chemistry(
    ec_numbers: list[str],
    catalytic_activities: list[str],
) -> dict:
    """Classify the bond type formed/broken by an enzyme using IUBMB EC rules.

    Uses the comprehensive EC mapping table plus reaction-text heuristics.
    Returns a dict with bond_types, reaction_direction, bond_context, and indicators.
    """
    bond_types: set[str] = set()
    directions: set[str] = set()
    contexts: list[str] = []
    indicators: list[str] = []

    for ec in ec_numbers:
        parts = ec.replace("-", "").split(".")
        if len(parts) < 2:
            continue
        try:
            ec_class = int(parts[0])
            ec_subclass = int(parts[1]) if parts[1] else None
        except (ValueError, IndexError):
            continue

        if ec_subclass is None:
            continue

        key = (ec_class, ec_subclass)
        if key in EC_BOND_TYPE_MAP:
            entry = EC_BOND_TYPE_MAP[key]
            bond_types.add(entry["bond_type"])
            directions.add(entry["direction"])
            contexts.append(entry["context"])

    for cat in catalytic_activities:
        cat_lower = cat.lower()

        if re.search(r"\bcoa\b", cat_lower):
            if "thioester" not in bond_types:
                bond_types.add("thioester")
            indicators.append("CoA in reaction")

        if re.search(
            r"\[acp\]|\[arcp\]|holo-\[|carrier\s+protein|pantetheine|\bACP\b",
            cat, re.IGNORECASE,
        ):
            if "thioester" not in bond_types:
                bond_types.add("thioester")
            indicators.append("carrier protein in reaction")

        if re.search(r"\bNH[34]\+?\b|\bammonia\b|\bammonium\b", cat, re.IGNORECASE):
            indicators.append("releases ammonia (primary amide substrate)")

    direction = None
    if len(directions) == 1:
        direction = directions.pop()
    elif directions:
        direction = sorted(directions)[0]

    return {
        "bond_types": sorted(bond_types),
        "reaction_direction": direction,
        "bond_context": contexts,
        "indicators": indicators,
    }


def extract_target_bond_type(target_reaction_type: str) -> dict:
    """Parse the target_reaction_type string for bond type and direction keywords."""
    target_lower = target_reaction_type.lower()
    target_bonds: set[str] = set()
    target_direction = None

    bond_keywords = [
        ("thioester", "thioester"),
        ("phosphoester", "phosphoester"),
        ("phosphodiester", "phosphodiester"),
        ("glycosidic", "glycosidic"),
        ("disulfide", "S-S"),
        ("peptide", "peptide"),
        ("amide", "amide"),
        ("ester", "ester"),
        ("ether", "ether"),
        ("c-c bond", "C-C"),
        ("c-n bond", "C-N"),
        ("c-o bond", "C-O"),
        ("c-s bond", "C-S"),
    ]
    for keyword, bond_type in bond_keywords:
        if keyword in target_lower:
            target_bonds.add(bond_type)

    _DIRECTION_KEYWORDS = {
        # Check explicit cleavage terms first.  Target descriptions commonly
        # say e.g. "hydrolysis (NOT synthesis)"; letting "synthesis" win
        # creates a false direction-mismatch warning for every hydrolase.
        "hydrolysis": "cleavage", "cleavage": "cleavage",
        "degradation": "cleavage", "hydrolase": "cleavage",
        "lysis": "cleavage", "decarboxylation": "cleavage",
        "deamination": "cleavage", "desulfurization": "cleavage",
        "dephosphorylation": "cleavage",
        "formation": "formation", "synthesis": "formation",
        "ligation": "formation", "condensation": "formation",
        "ligase": "formation",
        "transfer": "transfer", "transferase": "transfer",
        "phosphorylation": "transfer", "glycosylation": "transfer",
        "acylation": "transfer", "methylation": "transfer",
        "oxidation": "redox", "reduction": "redox",
        "oxidase": "redox", "reductase": "redox",
    }
    for kw, direction in _DIRECTION_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", target_lower):
            target_direction = direction
            break

    if not target_bonds and target_reaction_type.strip():
        logger.warning(
            "Target reaction type '%s' did not match any bond-type keywords. "
            "Bond-type validation will be skipped for this run.",
            target_reaction_type[:80],
        )

    return {
        "bond_types": sorted(target_bonds),
        "direction": target_direction,
    }


def filter_ec_prefixes_by_bond_type(
    ec_prefixes: list[str],
    reaction_text: str,
) -> list[str]:
    """Keep only EC prefixes whose bond type matches the target reaction.

    EC annotations are compatibility metadata, so this is a rejection filter:
    it discards prefixes whose bond chemistry or reaction direction contradicts
    the target. It never generates candidates. When the target reaction text
    yields no recognizable bond type, every prefix is returned unchanged.
    """
    target_info = extract_target_bond_type(reaction_text)
    target_bonds = set(target_info.get("bond_types", []))
    target_direction = target_info.get("direction")

    if not target_bonds:
        return ec_prefixes

    kept = []
    for prefix in ec_prefixes:
        parts = prefix.split(".")
        if len(parts) < 2:
            continue
        # Synthesize a fully qualified EC so the shared classifier can be
        # reused; only the class/subclass positions carry meaning here.
        synthetic_ec = f"{parts[0]}.{parts[1]}.0.0"
        chem = classify_reaction_chemistry([synthetic_ec], [])
        prefix_bonds = set(chem.get("bond_types", []))
        prefix_direction = chem.get("reaction_direction")

        if prefix_bonds & target_bonds:
            kept.append(prefix)
        elif target_direction and prefix_direction == target_direction:
            kept.append(prefix)

    if kept:
        logger.info("Bond-type filter kept EC prefixes: %s (from %s)", kept, ec_prefixes)
    else:
        logger.warning("Bond-type filter matched nothing; falling back to first 2 prefixes")
    return kept if kept else ec_prefixes[:2]


def check_bond_type_mismatch(
    enzyme_chem: dict,
    target_chem: dict,
) -> list[str]:
    """Compare enzyme's classified reaction chemistry against the target.

    Returns a list of weakness strings (may be empty if no mismatch detected).
    """
    warnings: list[str] = []

    enzyme_bonds = set(enzyme_chem.get("bond_types", []))
    target_bonds = set(target_chem.get("bond_types", []))
    enzyme_dir = enzyme_chem.get("reaction_direction")
    target_dir = target_chem.get("direction")

    if not enzyme_bonds or not target_bonds:
        return warnings

    expanded_enzyme = set(enzyme_bonds)
    expanded_target = set(target_bonds)
    if "amide" in expanded_target or "C-N" in expanded_target:
        expanded_target |= {"amide", "C-N"}
    if "amide" in expanded_enzyme or "C-N" in expanded_enzyme:
        expanded_enzyme |= {"amide", "C-N"}

    if not (expanded_enzyme & expanded_target):
        enzyme_str = ", ".join(sorted(enzyme_bonds))
        target_str = ", ".join(sorted(target_bonds))
        warnings.append(
            f"BOND TYPE MISMATCH: enzyme's classified bond type "
            f"({enzyme_str}) differs from target ({target_str})"
        )
        contexts = enzyme_chem.get("bond_context", [])
        if contexts:
            warnings[-1] += f" — EC context: {contexts[0]}"

    if "peptide" in enzyme_bonds and "amide" in target_bonds:
        warnings.append(
            "SUBSTRATE CLASS NOTE: enzyme is classified as a peptide bond "
            "hydrolase (EC 3.4.*). Peptide bond recognition depends on amino "
            "acid side chains and polypeptide backbone geometry that may be "
            "absent in the target substrate."
        )

    if enzyme_dir and target_dir and enzyme_dir != target_dir:
        hard_conflicts = {
            ("formation", "cleavage"),
            ("cleavage", "formation"),
        }
        if (enzyme_dir, target_dir) in hard_conflicts:
            warnings.append(
                f"REACTION DIRECTION MISMATCH: enzyme classified as "
                f"'{enzyme_dir}' but target requires '{target_dir}'"
            )

    return warnings


# --------------------------------------------------------------------------
# Property-based substrate comparison
# --------------------------------------------------------------------------

def compare_substrate_properties(
    target_smi: str,
    native_smi: str,
    compute_props_fn,
) -> dict:
    """Compare two substrates by physicochemical properties.

    Returns a dict with property-level comparison. Scores are 0-1 where 1 = identical.
    """
    try:
        t_props = compute_props_fn(target_smi)
        n_props = compute_props_fn(native_smi)
    except Exception:
        return {}

    mw_t, mw_n = t_props.molecular_weight, n_props.molecular_weight
    mw_sim = min(mw_t, mw_n) / max(mw_t, mw_n) if max(mw_t, mw_n) > 0 else 0.0

    cl_t, cl_n = t_props.longest_aliphatic_chain, n_props.longest_aliphatic_chain
    cl_diff = abs(cl_t - cl_n)
    cl_max = max(cl_t, cl_n, 1)
    cl_sim = max(0.0, 1.0 - cl_diff / cl_max)

    fg_t = set(t_props.functional_groups)
    fg_n = set(n_props.functional_groups)

    _fg_keys = (
        "num_carboxyl", "num_primary_amine", "num_secondary_amine",
        "num_tertiary_amine", "num_amide", "num_hydroxyl", "num_thiol",
        "num_ketone", "num_aldehyde", "num_ester", "num_ether",
        "num_epoxide", "num_phosphate", "num_sulfonate", "num_nitro",
        "num_nitrile", "num_imine", "num_guanidinium", "num_aromatic_ring",
    )
    fg_counts_t = {k.removeprefix("num_"): getattr(t_props, k, 0) for k in _fg_keys}
    fg_counts_n = {k.removeprefix("num_"): getattr(n_props, k, 0) for k in _fg_keys}

    # COUNT-aware functional-group similarity (Ruzicka / weighted Jaccard over the
    # per-group counts), NOT a binary set-Jaccard.  A set-Jaccard saturates at ~0.5
    # for almost every amide ligase — {carboxyl} vs {carboxyl, hydroxyl} and
    # {primary_amine} vs {primary_amine, secondary_amine} both score 0.5 — and since
    # the accommodation bottleneck takes the MIN across axes, that flat 0.5 capped
    # every candidate and erased the discriminating chain/size signal.  Counts keep
    # the discrimination: a diamine target vs a diamine native scores higher than vs
    # a single-amine amino acid, which is exactly the amine-partner difference the
    # ranking needs to separate a genuine diamine ligase from a small-substrate one.
    _num = sum(min(fg_counts_t[k], fg_counts_n[k]) for k in fg_counts_t)
    _den = sum(max(fg_counts_t[k], fg_counts_n[k]) for k in fg_counts_t)
    fg_sim = (_num / _den) if _den else 1.0

    logp_diff = abs(t_props.logP - n_props.logP)
    logp_sim = max(0.0, 1.0 - logp_diff / config.LOGP_PENALTY_DIVISOR)

    vol_t, vol_n = t_props.molecular_volume, n_props.molecular_volume
    vol_sim = min(vol_t, vol_n) / max(vol_t, vol_n) if max(vol_t, vol_n) > 0 else 0.0

    hbd_diff = abs(t_props.num_hbd - n_props.num_hbd)
    hba_diff = abs(t_props.num_hba - n_props.num_hba)

    rot_t, rot_n = t_props.num_rotatable_bonds, n_props.num_rotatable_bonds
    rot_diff = abs(rot_t - rot_n)

    tpsa_t, tpsa_n = t_props.tpsa, n_props.tpsa
    tpsa_sim = min(tpsa_t, tpsa_n) / max(tpsa_t, tpsa_n) if max(tpsa_t, tpsa_n) > 0 else 1.0

    charge_t = t_props.net_charge_ph8
    charge_n = n_props.net_charge_ph8

    arom_match = t_props.is_aromatic == n_props.is_aromatic
    cyclic_match = t_props.is_cyclic == n_props.is_cyclic

    return {
        "target_smiles": target_smi[:60],
        "native_smiles": native_smi[:60],
        "mw_target": round(mw_t, 1),
        "mw_native": round(mw_n, 1),
        "mw_similarity": round(mw_sim, 3),
        "volume_target": round(vol_t, 1),
        "volume_native": round(vol_n, 1),
        "volume_similarity": round(vol_sim, 3),
        "chain_length_target": cl_t,
        "chain_length_native": cl_n,
        "chain_length_similarity": round(cl_sim, 3),
        "rotatable_bonds_target": rot_t,
        "rotatable_bonds_native": rot_n,
        "rotatable_bonds_diff": rot_diff,
        "logP_similarity": round(logp_sim, 3),
        "tpsa_target": round(tpsa_t, 1),
        "tpsa_native": round(tpsa_n, 1),
        "tpsa_similarity": round(tpsa_sim, 3),
        "hbd_target": t_props.num_hbd,
        "hbd_native": n_props.num_hbd,
        "hbd_diff": hbd_diff,
        "hba_target": t_props.num_hba,
        "hba_native": n_props.num_hba,
        "hba_diff": hba_diff,
        "charge_ph8_target": charge_t,
        "charge_ph8_native": charge_n,
        "functional_groups_target": sorted(fg_t),
        "functional_groups_native": sorted(fg_n),
        "functional_group_similarity": round(fg_sim, 3),
        "functional_group_counts_target": fg_counts_t,
        "functional_group_counts_native": fg_counts_n,
        "aromaticity_match": arom_match,
        "cyclicity_match": cyclic_match,
    }
