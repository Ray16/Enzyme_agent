"""Shared family protocol compilation, handoff grouping, and ortholog deduplication."""
from __future__ import annotations

from collections import defaultdict
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from ..io_utils import file_digest
from ..modeling_reasoning.literature import LiteratureKnowledge
from ..modeling_reasoning.reaction import ReactionCoordinates
from ..modeling_reasoning.resolve_handoff import resolve_candidate

# Arm-directory names scanned for cross-run cache reuse and in-flight dedup.
# `cofolded` is the current convention (paired with `cofolded_restrained` for the
# pocket-guided step-2 arm); `family_cofold` is the legacy name, kept here so runs
# produced before the rename stay reusable. Only the unrestrained arm participates in
# reuse/dedup (the restrained arm carries a pocket constraint -> different geometry).
COFOLD_ARM_NAMES = ("cofolded", "family_cofold")
DEFAULT_SAMPLES = 50
DEFAULT_SEED = 42

# Generic structural-annotation pattern detecting obligate inter-subunit active
# sites from UniProt/PDB subunit or architecture evidence (never enzyme-specific).
_INTERFACIAL_ARCH_RE = re.compile(
    r"active\s+site.{0,40}(?:interface|between\s+(?:adjacent\s+)?subunits|inter[\s\-]?subunit)"
    r"|inter[\s\-]?subunit\s+active\s+site"
    r"|subunit[\s\-]+interface.{0,40}active\s+site",
    re.I,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


file_hash = file_digest


def target_key(smiles):
    from rdkit import Chem

    return digest([Chem.MolToSmiles(Chem.MolFromSmiles(value), isomericSmiles=True) for value in smiles])


def register_protocol(path, registry):
    """Cache an explicit shared target protocol, never candidate discoveries."""
    path = Path(path)
    registry = Path(registry)
    spec = json.loads(path.read_text())
    spec.pop("fleet_pilots", None)  # Operational cohort selectors never enter a reusable chemistry protocol.
    key = target_key([c["source_smiles"] for c in spec["components"]])
    directory = registry.parent / "protocols"
    directory.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(spec, indent=2) + "\n").encode()
    destination = directory / (hashlib.sha256(content).hexdigest() + ".json")
    if not destination.exists():
        destination.write_bytes(content)
    with state_lock(registry):
        data = json.loads(registry.read_text()) if registry.exists() else {}
        data[key] = {
            "path": str(destination.resolve()),
            "sha256": file_hash(destination),
            "scope": "User-supplied conditional shared protocol; not experimentally demonstrated activity.",
        }
        atomic_write_json(registry, data)


def adopt_registered_protocol(run, path):
    from enzyme_discovery_agent.models.target_review import target_contract

    registry = Path(run).parent / ".family_protocols/index.json"
    if not registry.exists():
        return False
    target = [s for s in (target_contract(run).get("substrate_smiles") or "").split(".") if s]
    if not target:
        return False
    reference = json.loads(registry.read_text()).get(target_key(target))
    if not reference:
        return False
    source = Path(reference["path"])
    if file_hash(source) != reference["sha256"]:
        raise ValueError("Registered protocol changed")
    atomic_write_json(path, json.loads(source.read_text()))
    return True


def compile_protocol(mode, specification):
    """Resolve both steps once for a mechanism/unit/target protocol."""
    coordinates = (
        ReactionCoordinates.load(specification["reaction_coordinates_path"])
        if specification.get("reaction_coordinates_path")
        else ReactionCoordinates.load()
    )
    key, coordinate, _ = coordinates.select(activation_mode=mode)
    if coordinate is None:
        raise ValueError("No shared reaction coordinate for " + str(mode))
    template = specification["mechanism_templates"][mode]
    metal = specification.get("metal_override") or template.get("metal")
    components = specification["components"]
    stages = {}
    knowledge = LiteratureKnowledge.load()
    for stage in coordinate["steps"]:
        index = stage["index"]
        partners = []
        directives = []
        for number, component in enumerate(components, 1):
            state = component["states"][mode][str(index)]
            included = state["include_in_model"]
            smiles = state["modeling_smiles"]
            role = state["role"]
            partners.append({
                "component_index": number,
                "source_smiles": component["source_smiles"],
                "modeling_smiles": smiles,
                "role": role,
                "include_in_model": included,
            })
            directives.append({
                "match": {"component_index": number},
                "include_in_model": included,
                "modeling_smiles": smiles,
                "role": role,
                "review_status": "reviewed",
                "origin": "user_protocol",
                "rationale": "Shared stage-specific representation from the supplied protocol.",
                **({"expected_formal_charge": state["expected_formal_charge"]} if state.get("expected_formal_charge") is not None else {}),
                **({"assay_pH": state["assay_pH"]} if state.get("assay_pH") is not None else {}),
                **({"allow_all_neutral_polyamine": bool(state["allow_all_neutral_polyamine"])} if "allow_all_neutral_polyamine" in state else {}),
            })
        nucleotide = template.get("nucleotide_ccd")
        candidate = {
            "accession": "shared-protocol",
            "evidence": {},
            "modeling_input": {
                "activation": {"status": "assigned", "activation_mode": mode, "nucleotide": nucleotide},
                "target_substrates": partners,
                "cofactors": copy.deepcopy(template.get("step_cofactors", {}).get(str(index), template.get("cofactors", []))),
            },
        }
        metal_resolution = (
            {
                "strategy": "explicit",
                "ccd": metal["metal"],
                "per_site_metal_count": metal["per_site_metal_count"],
                "uniprot_cross_check": False,
                "origin": "existing_mechanism_protocol",
                "rationale": metal["rationale"],
                **({"active_sites": int(specification["active_sites"])} if specification.get("active_sites") is not None else {}),
            }
            if metal
            else {"strategy": "omit"}
        )
        declaration = {
            "step": {"index": index, "mechanism": key, "chosen_because": specification["step_rationale"][str(index)]},
            "scoring_policy": "unrestrained",
            **({"assay_pH": specification["assay_pH"]} if specification.get("assay_pH") is not None else {}),
            "resolutions": {
                "enzyme_state": {
                    "chains": [{"source": "full_sequence"}],
                    "copies": specification.get("protein_copies", 1),
                    "state": "intact",
                    "origin": "shared_protocol",
                    "rationale": specification["assembly_basis"],
                },
                "substrates": directives,
                "catalytic_metal": metal_resolution,
            },
        }
        resolved = resolve_candidate(candidate, declaration, knowledge, step=index, mechanism_override=key, coordinates=coordinates)
        if resolved["status"] != "ready":
            raise ValueError(str(resolved["errors"]))
        stages[str(index)] = candidate["modeling_input"]
    return {
        "mechanism": mode,
        "coordinate": key,
        "steps": stages,
        "scope": "Shared conditional structural test, not established target activity.",
        "assembly_basis": specification["assembly_basis"],
        "setup_llm_calls": 0,
    }


def _family(data):
    value = data.get("interpro_family_id") or data.get("screening_family_id") or data.get("interpro_family")
    return json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else str(value or "unresolved")


def _canonical_family(data, row):
    """Group related InterPro sub-tags that share a canonical mechanism/fold family."""
    raw = _family(data)
    mech_family = str(row.get("mechanism_family") or data.get("mechanism_family") or "").strip()
    if mech_family:
        return mech_family, raw
    return raw, raw


def _load_discovery_handoff_index(root: Path) -> dict:
    """Read-only lookup of discovery's structural_modeling_handoff.json if present."""
    path = root / "structural_modeling_handoff.json"
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    by_acc = {}
    for cand in doc.get("candidates") or []:
        if isinstance(cand, dict) and cand.get("accession"):
            by_acc[cand["accession"]] = cand
    return by_acc


def _interfacial_assembly_override(raw_family, canonical_family, data, row, specification):
    """Return a minimal multi-chain override when generic evidence indicates an interfacial active site."""
    overrides = specification.get("family_overrides") or {}
    explicit = overrides.get(raw_family) or overrides.get(canonical_family) or {}
    if explicit.get("protein_copies") is not None:
        return {}
    if specification.get("disable_interfacial_assembly_override"):
        return {}

    interpro_ids = {raw_family}
    for key in ("interpro_family_id", "screening_family_id"):
        val = data.get(key)
        if isinstance(val, str) and val:
            interpro_ids.add(val)
        elif isinstance(val, list):
            interpro_ids.update(str(v) for v in val if v)
    fold_text = " ".join(
        str(x or "")
        for x in (
            data.get("interpro_family"),
            row.get("interpro_family"),
            row.get("mechanism_family"),
            data.get("mechanism_family"),
        )
    )
    for rule in specification.get("interfacial_assembly_rules") or []:
        if not isinstance(rule, dict):
            continue
        rule_ids = set(rule.get("family_ids") or [])
        rule_regex = rule.get("family_regex")
        if (rule_ids and (interpro_ids & rule_ids)) or (rule_regex and re.search(rule_regex, fold_text, re.I)):
            return {
                "protein_copies": int(rule.get("protein_copies", 2) or 2),
                "active_sites": int(rule.get("active_sites", 1) or 1),
                "assembly_basis": rule.get("assembly_basis") or (
                    "Minimal functional oligomer interface from protocol assembly rule "
                    f"({int(rule.get('protein_copies', 2) or 2)} protein copies, "
                    f"{int(rule.get('active_sites', 1) or 1)} modeled catalytic site)."
                ),
            }

    arch = data.get("architecture_and_tractability") or {}
    if isinstance(arch, dict) and arch.get("interfacial_active_site") is True:
        copies = int(arch.get("required_copies") or 2)
        return {
            "protein_copies": copies,
            "active_sites": 1,
            "assembly_basis": (
                f"Minimal functional interface override from candidate structural evidence "
                f"({copies} protein copies, 1 modeled catalytic site)."
            ),
        }
    evidence_text = " ".join(
        json.dumps(x) if isinstance(x, (dict, list)) else str(x or "")
        for x in (arch, data.get("subunit_structure"), data.get("function_comments"))
    )
    if _INTERFACIAL_ARCH_RE.search(evidence_text):
        return {
            "protein_copies": 2,
            "active_sites": 1,
            "assembly_basis": (
                "Minimal functional dimer interface override: candidate subunit/architecture evidence "
                "indicates an inter-subunit active site (2 protein copies, 1 modeled catalytic site)."
            ),
        }
    return {}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def deduplicate_handoff_orthologs(handoff, candidates=None, *, identity_threshold: float | None = 0.85):
    """Collapse near-identical orthologs (>= identity_threshold) within each family group."""
    if identity_threshold is None or identity_threshold <= 0:
        handoff.setdefault("collapsed_orthologs", [])
        return handoff
    from enzyme_discovery_agent.models.ortholog import filter_highly_similar_orthologs

    candidates = candidates or {}
    collapsed_all = list(handoff.get("collapsed_orthologs") or [])
    seen_collapsed = {c.get("accession") for c in collapsed_all if isinstance(c, dict)}
    for group in handoff.get("families") or []:
        members = list(group.get("members") or [])
        members.sort(
            key=lambda m: (
                0 if m.get("is_swissprot") else 1,
                0 if m.get("recorded_activities") else 1,
                int(m.get("rank") or 999999),
                m.get("accession") or "",
            )
        )
        kept, collapsed = filter_highly_similar_orthologs(
            members,
            identity_threshold=float(identity_threshold),
            seq_getter=lambda m: (candidates.get(m.get("accession")) or {}).get("sequence") or m.get("_sequence"),
            family_getter=lambda m: group.get("family_id") or "same_family",
        )
        group["members"] = kept
        for item in collapsed:
            if item.get("accession") not in seen_collapsed:
                seen_collapsed.add(item.get("accession"))
                collapsed_all.append(item)
    handoff["collapsed_orthologs"] = collapsed_all
    return handoff


def build_handoff(run_dir, specification, *, registry_loader=None):
    """Group published candidates; direct contradictory evidence always splits groups."""
    root = Path(run_dir)
    loader = registry_loader or load_candidate_registry
    registry = loader(root, strict=True)["candidates"]
    discovery_handoff = _load_discovery_handoff_index(root)
    ranking_path = root / "ranked_candidates.csv"
    rows = list(csv.DictReader(ranking_path.open())) if ranking_path.is_file() else []
    assignments = []
    known = defaultdict(set)
    coordinates = (
        ReactionCoordinates.load(specification["reaction_coordinates_path"])
        if specification.get("reaction_coordinates_path")
        else ReactionCoordinates.load()
    )
    for row in rows:
        meta = (registry.get(row["accession"]) or {}).get("metadata") or {}
        dh_cand = discovery_handoff.get(row["accession"]) or {}
        dh_ev = dh_cand.get("evidence") or {}
        data = {**(meta.get("evaluation_data") or {}), **meta}
        if dh_ev.get("architecture_and_tractability") and not data.get("architecture_and_tractability"):
            data["architecture_and_tractability"] = dh_ev["architecture_and_tractability"]
        if dh_cand.get("sequence") and not data.get("sequence"):
            data["sequence"] = dh_cand["sequence"]
        canonical_fam, raw_fam = _canonical_family(data, row)
        text = " ".join(data.get("catalytic_activities") or [])
        matched = [
            rule["mode"]
            for rule in specification.get("assignment_rules", [])
            if all(re.search(pattern, text, re.I) for pattern in rule.get("required", []))
            and not any(re.search(pattern, text, re.I) for pattern in rule.get("excluded", []))
            and rule.get("required")
        ]
        basis = "recorded_reaction"
        if len(set(matched)) == 1:
            mode = matched[0]
            known[canonical_fam].add(mode)
            known[raw_fam].add(mode)
        else:
            mode = ((data.get("modeling_input") or {}).get("activation") or {}).get("activation_mode")
            basis = "source_activation_annotation"
            if mode not in specification["mechanism_templates"]:
                label = str(data.get("interpro_family") or row.get("mechanism_family") or "")
                modes = {
                    rule["mode"]
                    for rule in specification.get("assignment_rules", [])
                    if rule.get("family_regex") and re.search(rule["family_regex"], label, re.I)
                }
                mode = next(iter(modes)) if len(modes) == 1 else None
                basis = "conditional_family_transfer"
            if not mode:
                ids = [raw_fam] if raw_fam.startswith("IPR") else []
                _, coordinate, _ = coordinates.select(family_ids=ids)
                modes = (coordinate or {}).get("keys", {}).get("activation_mode", [])
                mode = modes[0] if len(modes) == 1 else None
                basis = "conditional_family_coordinate"
        assignments.append((row, data, canonical_fam, raw_fam, mode, basis, dh_ev))

    protocols = {}
    groups = {}
    pending = []
    invalid = {}
    for row, data, canonical_fam, raw_fam, mode, basis, dh_ev in assignments:
        if not mode:
            fam_modes = known[canonical_fam] or known[raw_fam]
            if len(fam_modes) == 1:
                mode = next(iter(fam_modes))
                basis = "conditional_family_member_transfer"
        is_sp = _as_bool(row.get("is_swissprot")) if row.get("is_swissprot") not in (None, "") else _as_bool(data.get("is_swissprot"))
        ta_summary = dh_ev.get("target_assessment_summary") or {}
        if not ta_summary and isinstance(data.get("target_assessment"), dict):
            ta = data["target_assessment"]
            ta_summary = {
                k: ta.get(k)
                for k in ("decision", "evidence_level", "mechanism_fit", "pocket_engineering_outlook")
                if ta.get(k) is not None
            }
        member = {
            "accession": row["accession"],
            "enzyme_name": row.get("protein_name") or row.get("enzyme_name") or data.get("protein_name"),
            "rank": int(row.get("rank") or len(rows)),
            "assignment_basis": basis,
            "hypothesis_class": "native_mechanism_match" if basis == "recorded_reaction" else "exploratory_transfer",
            "is_swissprot": is_sp,
            "raw_family_id": raw_fam,
            "recorded_activities": data.get("catalytic_activities") or [],
            **({"architecture_and_tractability": data["architecture_and_tractability"]} if data.get("architecture_and_tractability") else {}),
            **({"target_assessment_summary": ta_summary} if ta_summary else {}),
            **({"_sequence": data["sequence"]} if isinstance(data.get("sequence"), str) and data["sequence"] else {}),
        }
        if mode not in specification["mechanism_templates"]:
            pending.append({
                k: v
                for k, v in {**member, "family_id": canonical_fam, "reason": "No applicable shared mechanism protocol; resolve this family once."}.items()
                if not k.startswith("_")
            })
            continue
        overrides = specification.get("family_overrides") or {}
        explicit_override = overrides.get(raw_fam) or overrides.get(canonical_fam) or {}
        auto_override = _interfacial_assembly_override(raw_fam, canonical_fam, data, row, specification)
        shared = {**specification, **auto_override, **explicit_override}
        member["protein_copies"] = int(shared.get("protein_copies", 1) or 1)
        protocol_id = digest({"mode": mode, "specification": shared})[:20]
        if protocol_id not in protocols and protocol_id not in invalid:
            try:
                protocols[protocol_id] = compile_protocol(mode, shared)
            except (KeyError, ValueError) as exc:
                invalid[protocol_id] = str(exc)
        if protocol_id in invalid:
            pending.append({
                k: v
                for k, v in {**member, "family_id": canonical_fam, "reason": "Shared protocol needs review: " + invalid[protocol_id]}.items()
                if not k.startswith("_")
            })
            continue
        group_id = canonical_fam + ":" + mode + ":c" + str(member["protein_copies"])
        group = groups.setdefault(
            group_id,
            {"family_id": canonical_fam, "protocol_id": protocol_id, "protein_copies": member["protein_copies"], "members": []},
        )
        group["members"].append(member)

    handoff = {
        "schema_version": "family-cofold-1",
        "discovery_run_id": root.name,
        "source_run": str(root.resolve()),
        "specification_sha256": digest(specification),
        "protocols": protocols,
        "families": list(groups.values()),
        "pending_families": pending,
        "collapsed_orthologs": [],
        "ranked_candidates": len(rows),
        "setup_llm_calls": 0,
        **({"expected_predictions": int(specification["expected_predictions"])} if specification.get("expected_predictions") is not None else {}),
        **({"seed": int(specification["seed"])} if specification.get("seed") is not None else {}),
    }
    threshold = specification.get("ortholog_identity_threshold", 0.85)
    deduplicate_handoff_orthologs(handoff, identity_threshold=threshold)
    for group in handoff["families"]:
        group["members"] = [{k: v for k, v in m.items() if not k.startswith("_")} for m in group["members"]]
    return handoff


def write_family_handoff_markdown(handoff, path):
    """Write a concise, human-readable Markdown companion to family_handoff.json."""
    path = Path(path)
    families = handoff.get("families") or []
    protocols = handoff.get("protocols") or {}
    pending = handoff.get("pending_families") or []
    collapsed = handoff.get("collapsed_orthologs") or []
    total_members = sum(len(g.get("members") or []) for g in families)

    def _step_desc(step_model):
        if not isinstance(step_model, dict):
            return "-"
        cofs = ", ".join(f"{c['ccd']}×{c['count']}" for c in (step_model.get("cofactors") or []))
        sub_parts = []
        for s in step_model.get("target_substrates") or []:
            if not s.get("include_in_model", True):
                continue
            role_txt = s.get("role") or "substrate"
            rep = s.get("modeling_representation") or {}
            tags = []
            if rep.get("formal_charge") is not None:
                q = int(rep["formal_charge"])
                tags.append(f"q={q:+d}" if q else "q=0")
            stereo = rep.get("stereochemistry") or {}
            if stereo.get("status") == "specified":
                cips = "".join(c.get("cip", "") for c in (stereo.get("assigned_centers") or []))
                if cips:
                    tags.append(f"CIP={cips}")
            sub_parts.append(f"{role_txt}[{','.join(tags)}]" if tags else role_txt)
        subs = ", ".join(sub_parts)
        return f"{subs} ({cofs})" if cofs else subs

    lines = [
        "# Family Structural-Modeling Handoff",
        "",
        f"- **Discovery run**: `{handoff.get('discovery_run_id', '-')}`",
        f"- **Ranked candidates**: **{handoff.get('ranked_candidates', 0)}** → "
        f"**{total_members}** scheduled across **{len(families)}** family group(s) · "
        f"**{len(collapsed)}** near-identical ortholog(s) collapsed · "
        f"**{len(pending)}** pending family resolution",
        f"- **Setup LLM calls**: `{handoff.get('setup_llm_calls', 0)}`",
        "",
        "## Resolved Family Protocols",
        "",
        "| Family | Mechanism | Copies | Step 1 Setup | Step 2 Setup | Members |",
        "| --- | --- | :---: | --- | --- | ---: |",
    ]
    for group in families:
        proto = protocols.get(group.get("protocol_id")) or {}
        steps = proto.get("steps") or {}
        copies = group.get("protein_copies", 1)
        copy_label = f"{copies} (dimer)" if copies == 2 else str(copies)
        lines.append(
            f"| {group.get('family_id')} | `{proto.get('mechanism', '-')}` | {copy_label} | "
            f"{_step_desc(steps.get('1'))} | {_step_desc(steps.get('2'))} | {len(group.get('members') or [])} |"
        )
    lines.append("")

    lines.extend(["## Scheduled Members by Family", ""])
    for group in families:
        proto = protocols.get(group.get("protocol_id")) or {}
        lines.append(f"### {group.get('family_id')} (`{proto.get('mechanism', '-')}`, {group.get('protein_copies', 1)} chain(s))")
        if proto.get("assembly_basis"):
            lines.append(f"> {proto['assembly_basis']}")
        lines.append("")
        for m in group.get("members") or []:
            sp = "Swiss-Prot" if m.get("is_swissprot") else "TrEMBL"
            hyp = m.get("hypothesis_class") or m.get("assignment_basis") or "-"
            lines.append(
                f"- **#{m.get('rank', '-')} {m.get('enzyme_name') or 'Unresolved'} (`{m.get('accession')}`)** "
                f"— {sp}, `{hyp}`"
            )
        lines.append("")

    if collapsed:
        lines.extend([
            "## Collapsed Near-Identical Orthologs (Retained Representative Modeled)",
            "",
            "| Collapsed Enzyme (Accession) | Representative | Seq Identity | Reason |",
            "| --- | --- | ---: | --- |",
        ])
        for c in collapsed:
            ident = f"{float(c['sequence_identity']):.1%}" if c.get("sequence_identity") is not None else "-"
            lines.append(
                f"| {c.get('protein_name') or 'Unresolved'} (`{c.get('accession')}`) | "
                f"`{c.get('representative_accession')}` | {ident} | {c.get('reason', '-')} |"
            )
        lines.append("")

    if pending:
        lines.extend(["## Pending / Unresolved Families (Not Yet Cofolded)", ""])
        for p in pending:
            lines.append(
                f"- **{p.get('enzyme_name') or 'Unresolved'} (`{p.get('accession')}`)** "
                f"[{p.get('family_id', '-')}]: {p.get('reason', '-')}"
            )
        lines.append("")

    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
