"""Agent-invoked CPU evaluation tools for saturation campaigns and design contacts.

Closes the design -> inspect -> evaluate loop so the design agent can inspect
landed saturation outputs (via ``saturation.results`` / ``saturation.evaluate``)
and audit protein-donor / ligand-acceptor contacts on PDB or mmCIF structures
(via ``contact_analysis``) without touching a GPU.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from enzyme_core.state_store import atomic_write_json
from enzyme_discovery_agent.tool_registry import tool

from ..contact_analysis import (
    Atom,
    ContactPolicy,
    TargetGroup,
    collapse_for_display,
    distinct_residue_count,
    evaluate_contacts,
)
from ..saturation.evaluate import evaluate as evaluate_saturation_manifest
from ..saturation.results import aggregate, discover, long_rows
from ..structure import load_structure


@tool(
    description=(
        "Inspect and summarize a saturation-mutagenesis campaign directory on disk: "
        "discovers landed models across layouts without double-counting, reports "
        "completion counts (n_models and complete flag), provenance confidence "
        "aggregates, and any manifest-based catalytic geometry evaluation. "
        "CPU-only and fast. Raw predictor confidences are provenance, not activity "
        "measurements; incomplete variants are never treated as low-scoring."
    ),
    category="structure",
    parallel_safe=True,
    params={
        "results_dir": {
            "type": "string",
            "description": (
                "Root directory of the saturation campaign (containing results/ "
                "and optionally manifest.json / evaluation.json)."
            ),
        },
        "modes": {
            "type": "string",
            "description": (
                "Optional comma-separated condition labels (e.g. 'apo,holo') "
                "used as path segments under results_dir."
            ),
            "default": "",
        },
        "value_key": {
            "type": "string",
            "description": (
                "Numeric confidence provenance column to aggregate per (id, mode), "
                "e.g. 'iptm', 'ligand_iptm', 'complex_plddt', 'confidence_score'."
            ),
            "default": "iptm",
        },
        "expect_models": {
            "type": "integer",
            "description": (
                "Expected models per (variant, mode) for marking completeness. "
                "0 omits the completeness flag unless manifest.json specifies n_samples."
            ),
            "default": 0,
        },
        "run_geometry_evaluation": {
            "type": "boolean",
            "description": (
                "If true and both manifest.json and evaluation.json exist in "
                "results_dir, run the manifest catalytic-geometry evaluator and "
                "return its status and top rows."
            ),
            "default": False,
        },
        "max_rows": {
            "type": "integer",
            "description": "Maximum aggregated rows to return.",
            "default": 50,
        },
    },
)
def summarize_saturation_campaign(
    results_dir: str,
    modes: str = "",
    value_key: str = "iptm",
    expect_models: int = 0,
    run_geometry_evaluation: bool = False,
    max_rows: int = 50,
) -> dict:
    """Summarize landed saturation predictions and optional geometry evaluation."""
    root = Path(results_dir).expanduser()
    if not root.is_dir():
        return {
            "status": "error",
            "summary": f"No such saturation campaign directory: {root}",
            "data": None,
            "suggestions": ["Pass the campaign output_dir or results directory."],
        }

    mode_tuple = tuple(m.strip() for m in modes.split(",") if m.strip())
    manifest_path = root / "manifest.json"
    known_ids: set[str] | None = None
    expect: int | None = expect_models if expect_models > 0 else None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            if isinstance(manifest.get("jobs"), list) and manifest["jobs"]:
                known_ids = {j["id"] for j in manifest["jobs"] if isinstance(j, dict) and "id" in j}
            if expect is None and isinstance(manifest.get("n_samples"), int) and manifest["n_samples"] > 0:
                expect = manifest["n_samples"]
        except (OSError, json.JSONDecodeError):
            manifest = None
    else:
        manifest = None

    camp = discover(root, modes=mode_tuple, known_ids=known_ids)
    rows = long_rows(camp)
    agg = aggregate(rows, value_key=value_key, expect=expect)

    geom_status = None
    geom_rows: list[dict] = []
    if run_geometry_evaluation and manifest_path.is_file() and (root / "evaluation.json").is_file():
        try:
            geom_status = evaluate_saturation_manifest(manifest_path)
            metrics_csv = root / "variant_metrics.csv"
            if metrics_csv.is_file():
                with metrics_csv.open(newline="") as fh:
                    geom_rows = list(csv.DictReader(fh))[:max_rows]
        except (OSError, ValueError, KeyError) as exc:
            geom_status = {"error": str(exc)}
    elif (root / "evaluation_status.json").is_file():
        try:
            geom_status = json.loads((root / "evaluation_status.json").read_text())
        except (OSError, json.JSONDecodeError):
            geom_status = None

    n_complete = (
        sum(1 for r in agg if r.get("complete") == 1)
        if expect is not None
        else None
    )
    summary = (
        f"Discovered {len(camp.models)} model(s) across {len(agg)} group(s) "
        f"in {root.name}"
    )
    if n_complete is not None:
        summary += f" ({n_complete}/{len(agg)} complete at n={expect})"
    if camp.duplicates:
        summary += f"; {camp.duplicates} duplicate(s) skipped"
    if camp.unknown_ids:
        summary += f"; {len(camp.unknown_ids)} unknown id(s) excluded"

    return {
        "status": "success",
        "summary": summary,
        "data": {
            "results_dir": str(root),
            "n_models": len(camp.models),
            "n_groups": len(agg),
            "n_complete_groups": n_complete,
            "expect_models": expect,
            "modes": list(camp.modes),
            "duplicates": camp.duplicates,
            "unknown_ids": list(camp.unknown_ids)[:20],
            "value_key": value_key,
            "summary_rows": agg[:max_rows],
            "geometry_evaluation": geom_status,
            "geometry_rows": geom_rows,
        },
        "suggestions": [
            "Confidence aggregates are predictor provenance, not activity scores.",
            "Check n_models and complete before comparing variants; verify negative controls.",
        ],
    }


def _structure_to_atoms(structure_path: Path) -> list[Atom]:
    st = load_structure(structure_path)
    atoms: list[Atom] = []
    for r in st.residues:
        resi = f"{r.resnum}{r.icode}"
        for idx, xyz in enumerate(r.xyz):
            name = r.atom_names[idx] if idx < len(r.atom_names) else f"A{idx}"
            elem = r.elements[idx] if idx < len(r.elements) else name[:1]
            atoms.append(Atom(
                chain=r.chain, residue=r.resname, resi=resi,
                name=name, element=elem,
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            ))
    for c in st.components:
        resi = str(c.resnum)
        for idx, xyz in enumerate(c.xyz):
            name = c.atom_names[idx] if idx < len(c.atom_names) else f"H{idx}"
            elem = c.elements[idx] if idx < len(c.elements) else name[:1]
            atoms.append(Atom(
                chain=c.chain, residue=c.resname, resi=resi,
                name=name, element=elem,
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            ))
    return atoms


@tool(
    description=(
        "Audit hydrogen-free protein-sidechain donor to ligand-acceptor contacts "
        "in a PDB or mmCIF structure using the versioned contact policy. "
        "Classifies compatible polar and potential ionic contacts, quarantines "
        "clashing or anomalously short residue-group pairs, and separates "
        "ambiguous His/Cys protonation states. CPU-only and fast."
    ),
    category="structure",
    parallel_safe=True,
    params={
        "structure_path": {
            "type": "string",
            "description": "Path to a PDB or mmCIF structure file.",
        },
        "target_groups": {
            "type": "array",
            "description": (
                "Ligand acceptor groups from the chemical graph. Each object needs: "
                "'group_id' (string), 'chain' (ligand chain id), 'resi' (residue number "
                "as string), 'acceptors' (list of exact acceptor atom names, e.g. ['O1','O2']), "
                "'formal_charge' (integer, e.g. -1 for carboxylate, 0 for neutral), and "
                "'provenance' (non-empty string explaining chemical origin)."
            ),
            "items": {"type": "object"},
        },
        "protein_chain": {
            "type": "string",
            "description": "Protein chain id to audit.",
            "default": "A",
        },
        "polar_max_A": {
            "type": "number",
            "description": "Maximum heavy-atom distance for polar donor-acceptor contacts (A).",
            "default": 3.5,
        },
        "ionic_max_A": {
            "type": "number",
            "description": "Maximum heavy-atom distance for potential ionic contacts (A).",
            "default": 4.0,
        },
        "audit_max_A": {
            "type": "number",
            "description": "Outer distance cutoff for reporting audited pairs (A).",
            "default": 4.5,
        },
        "minimum_contact_A": {
            "type": "number",
            "description": "Minimum non-anomalous contact distance (A).",
            "default": 2.4,
        },
    },
)
def audit_design_contacts(
    structure_path: str,
    target_groups: list[dict],
    protein_chain: str = "A",
    polar_max_A: float = 3.5,
    ionic_max_A: float = 4.0,
    audit_max_A: float = 4.5,
    minimum_contact_A: float = 2.4,
) -> dict:
    """Run versioned heavy-atom donor/acceptor contact analysis on a design."""
    path = Path(structure_path).expanduser()
    if not path.is_file():
        return {
            "status": "error",
            "summary": f"No such structure file: {path}",
            "data": None,
            "suggestions": ["Provide an existing PDB or mmCIF structure."],
        }
    if not target_groups:
        return {
            "status": "error",
            "summary": "target_groups must be a non-empty list of acceptor group definitions.",
            "data": None,
            "suggestions": ["Specify chain, resi, acceptors, formal_charge, and provenance."],
        }

    try:
        policy = ContactPolicy(
            minimum_contact_A=float(minimum_contact_A),
            polar_max_A=float(polar_max_A),
            ionic_max_A=float(ionic_max_A),
            audit_max_A=float(audit_max_A),
        )
        groups = [
            TargetGroup(
                group_id=str(g["group_id"]),
                chain=str(g["chain"]),
                resi=str(g["resi"]),
                acceptors=tuple(str(a) for a in g["acceptors"]),
                formal_charge=int(g["formal_charge"]),
                provenance=str(g["provenance"]),
            )
            for g in target_groups
        ]
        atoms = _structure_to_atoms(path)
        contacts = evaluate_contacts(atoms, groups, policy=policy, protein_chain=protein_chain)
        collapsed = collapse_for_display(contacts)
        n_residues = distinct_residue_count(contacts)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "status": "error",
            "summary": f"Contact audit failed: {exc}",
            "data": None,
            "suggestions": [
                "Verify chain, resi, and exact acceptor atom names against the structure.",
            ],
        }

    accepted = [c for c in contacts if c["accepted"]]
    conditional = [c for c in contacts if c["conditional"]]
    rejected = [c for c in contacts if not c["accepted"] and not c["conditional"]]
    return {
        "status": "success",
        "summary": (
            f"{len(accepted)} accepted contact(s) from {n_residues} distinct residue(s), "
            f"{len(conditional)} conditional/quarantined, {len(rejected)} rejected "
            f"(policy {policy.version})."
        ),
        "data": {
            "structure_path": str(path),
            "policy": policy.to_dict(),
            "distinct_accepted_residues": n_residues,
            "n_accepted": len(accepted),
            "n_conditional": len(conditional),
            "n_rejected": len(rejected),
            "collapsed_contacts": collapsed,
            "contacts": contacts,
        },
        "suggestions": [
            "Heavy-atom contacts are geometric compatibility checks, not hydrogen-bond energies.",
            "Inspect conditional contacts for unresolved His/Cys states or compressed geometries.",
        ],
    }


@tool(
    description=(
        "Record an auditable enzyme-design strategy decision BEFORE launching GPU "
        "generation or saturation campaigns. Captures the chosen strategy, backend, "
        "scientific rationale, residue-by-residue motif or mutable-set specification, "
        "negative controls, and falsification criteria to agent_state/design_strategy.json. "
        "Revisions append to the decision log rather than overwriting prior choices."
    ),
    category="design",
    parallel_safe=False,
    injected_params=["run_dir"],
    params={
        "strategy": {
            "type": "string",
            "enum": ["generative", "redesign", "both"],
            "description": "Chosen design strategy family.",
        },
        "backend": {
            "type": "string",
            "description": (
                "Planned tool/backend (e.g. 'rfd4-proteina', 'alphaprotein-novo', "
                "'disco', 'genzyme', 'saturation_redesign', or 'combined')."
            ),
        },
        "rationale": {
            "type": "string",
            "description": (
                "Why this strategy fits the target chemistry, available scaffold/handoff "
                "evidence, and why alternative strategies were not chosen."
            ),
        },
        "motif_or_mutable_spec": {
            "type": "string",
            "description": (
                "Residue-by-residue catalytic motif roles (for generative) or pocket/"
                "interface mutable-set definition and cutoff (for redesign)."
            ),
        },
        "falsification_criteria": {
            "type": "string",
            "description": (
                "Explicit geometric, refolding, or contact criteria that would reject "
                "the resulting designs."
            ),
        },
        "negative_controls": {
            "type": "string",
            "description": (
                "Optional negative controls (e.g. catalytic knockouts, apo/native "
                "controls, or decoy substrates)."
            ),
            "default": "",
        },
    },
)
def declare_design_strategy(
    strategy: str,
    backend: str,
    rationale: str,
    motif_or_mutable_spec: str,
    falsification_criteria: str,
    negative_controls: str = "",
    run_dir: str = "",
) -> dict:
    """Persist a structured, versioned strategy decision record before compute."""
    if strategy not in {"generative", "redesign", "both"}:
        return {
            "status": "error",
            "summary": f"Invalid strategy {strategy!r}; expected 'generative', 'redesign', or 'both'.",
            "data": None,
            "suggestions": ["Choose one of 'generative', 'redesign', or 'both'."],
        }
    missing = [
        name
        for name, val in (
            ("backend", backend),
            ("rationale", rationale),
            ("motif_or_mutable_spec", motif_or_mutable_spec),
            ("falsification_criteria", falsification_criteria),
        )
        if not isinstance(val, str) or not val.strip()
    ]
    if missing:
        return {
            "status": "error",
            "summary": f"declare_design_strategy requires non-empty fields: {', '.join(missing)}.",
            "data": None,
            "suggestions": ["State concrete scientific rationale, motif/mutable spec, and falsification criteria."],
        }

    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "backend": backend.strip(),
        "rationale": rationale.strip(),
        "motif_or_mutable_spec": motif_or_mutable_spec.strip(),
        "falsification_criteria": falsification_criteria.strip(),
        "negative_controls": (negative_controls or "").strip(),
    }

    ledger_path: Path | None = None
    revision = 1
    if run_dir:
        ledger_path = Path(run_dir) / "agent_state" / "design_strategy.json"
        existing: list[dict] = []
        if ledger_path.is_file():
            try:
                loaded = json.loads(ledger_path.read_text())
                if isinstance(loaded, dict) and isinstance(loaded.get("decisions"), list):
                    existing = loaded["decisions"]
            except (OSError, json.JSONDecodeError):
                existing = []
        revision = len(existing) + 1
        entry["revision"] = revision
        existing.append(entry)
        atomic_write_json(ledger_path, {"current": entry, "decisions": existing})
    else:
        entry["revision"] = revision

    return {
        "status": "success",
        "summary": (
            f"Recorded design strategy revision {revision}: strategy={strategy}, "
            f"backend={entry['backend']}."
        ),
        "data": {
            **entry,
            "ledger_path": str(ledger_path) if ledger_path else None,
        },
        "suggestions": [
            "Check backend readiness before launching GPU jobs.",
            "Keep falsification criteria and negative controls intact when evaluating outputs.",
        ],
    }
