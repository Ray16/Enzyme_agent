"""Promotion bridges connecting family-cofold -> MD/Design and Design -> Modeling validation.

Encodes the verified operational requirements from live Boltz-2 -> OpenMM MD testing:
1. Durable cofactor assets (``barrier/assets/ATP.lib`` and ``ATP.frcmod``).
2. Audited GAFF2 Li-Merz 12-6-4 polarizability overrides (``c5``, ``nv``, ``nz``).
3. Explicit two-environment split: AmberTools prep interpreter vs. CUDA OpenMM MD interpreter.
4. Mandatory PROPKA protonation-audit gate before launching paired microstate arms.
5. Explicit ``ReactionSpec`` binding and SHA-256 provenance on every promoted pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from enzyme_core.state_store import atomic_write_json

from ..barrier.parameters import DEFAULT_GAFF2_POLARIZABILITY_OVERRIDES
from ..barrier.prep_cofold import asset_dir
from ..validation import file_digest

DEFAULT_PREP_PYTHON = os.environ.get(
    "ENZYME_MD_PREP_PYTHON",
    "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/bin/python",
)
DEFAULT_MD_PYTHON = os.environ.get(
    "ENZYME_MD_CUDA_PYTHON",
    "/nfs/lambda_stor_01/homes/rzhu/Tool/envs/processivity-md-20261106/bin/python",
)


def _select_best_pose(complete_json: Path) -> dict[str, Any]:
    """Select the closest-attack-distance identity-passing pose from a complete.json."""
    payload = json.loads(complete_json.read_text())
    poses = (payload.get("analysis") or {}).get("poses") or []
    if not poses:
        raise ValueError(f"No poses recorded in {complete_json}")
    valid = [p for p in poses if not p.get("identity_issues")]
    pool = valid or poses
    best = min(pool, key=lambda p: float(p.get("distance_A", float("inf"))))
    pdb_path = Path(best["path"]).resolve()
    return {
        "pose_path": str(pdb_path),
        "pose_sha256": file_digest(pdb_path) if pdb_path.is_file() else None,
        "distance_A": float(best["distance_A"]),
        "angle_deg": float(best["angle_deg"]),
        "identity_pass": not bool(best.get("identity_issues")),
    }


def promote_family_cofold(
    family_dir: str | Path,
    *,
    top_k: int = 3,
    step: int = 1,
    arm: str = "unrestrained",
    md_spec: str | Path | None = None,
    ph: float = 8.0,
    prep_python: str = DEFAULT_PREP_PYTHON,
    md_python: str = DEFAULT_MD_PYTHON,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Promote top-K completed family-cofold candidates to MD prep and Design handoffs."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    root = Path(family_dir).resolve()
    report_path = root / "analysis" / "family_modeling_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Missing {report_path}; run structural_modeling_agent.family.analysis.refresh() first."
        )
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != "family-cofold-1.0":
        raise ValueError(f"Expected family-cofold-1.0 report at {report_path}")

    matching: list[dict[str, Any]] = []
    for cand in report.get("candidates", []):
        for st in cand.get("steps", []):
            if st.get("step") != step or st.get("analysis_arm", "unrestrained") != arm:
                continue
            angle = st.get("weighted_attack_angle_deg")
            if angle is None:
                continue
            matching.append(
                {
                    "accession": cand["accession"],
                    "enzyme_name": cand.get("enzyme_name"),
                    "family_id": cand.get("family_id"),
                    **st,
                }
            )

    matching.sort(
        key=lambda row: (-float(row["weighted_attack_angle_deg"]), float(row.get("harmonic_distance_A", 999.0)))
    )
    selected = matching[:top_k]

    dest = Path(out_dir).resolve() if out_dir else root / "promotion"
    dest.mkdir(parents=True, exist_ok=True)
    cofactor_assets = asset_dir().resolve()
    spec_str = str(Path(md_spec).resolve()) if md_spec else None
    amber_home = str(Path(prep_python).resolve().parent.parent)
    pol_path = str(Path(amber_home) / "dat" / "leap" / "parm" / "lj_1264_pol.dat")

    promoted_md: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, 1):
        result_path = Path(row["result_path"]).resolve()
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing family-cofold result artifact: {result_path}")
        if row.get("result_sha256") and file_digest(result_path) != row["result_sha256"]:
            raise ValueError(f"Result artifact digest mismatch for {row['accession']}: {result_path}")
        best_pose = _select_best_pose(result_path)
        item = {
            "rank": rank,
            "accession": row["accession"],
            "enzyme_name": row.get("enzyme_name"),
            "family_id": row.get("family_id"),
            "step": step,
            "analysis_arm": arm,
            "weighted_attack_angle_deg": row["weighted_attack_angle_deg"],
            "harmonic_distance_A": row.get("harmonic_distance_A"),
            "near_attack_fraction": row.get("near_attack_fraction"),
            "holo_seating_status": row.get("holo_seating_status"),
            "source_result": {"path": str(result_path), "sha256": file_digest(result_path)},
            "selected_pose": best_pose,
            "md_setup": {
                "ph": ph,
                "reaction_spec": spec_str,
                "cofactor_asset_dir": str(cofactor_assets),
                "polarizabilities_dat": pol_path,
                "polarizability_overrides": dict(DEFAULT_GAFF2_POLARIZABILITY_OVERRIDES),
                "prep_python": prep_python,
                "md_python": md_python,
                "protonation_audit": True,
                "paired_microstate_policy": (
                    "Inspect PROPKA protonation_audit findings before launching MD; "
                    "run paired microstate arms when buried active-site residues are flagged."
                ),
            },
        }
        promoted_md.append(item)

    md_manifest_path = dest / f"md_promotion_step{step}_{arm}.json"
    md_manifest = {
        "schema_version": "md-promotion-1.0",
        "kind": "family_cofold_md_promotion",
        "source_report": {"path": str(report_path), "sha256": file_digest(report_path)},
        "source_discovery_run": report.get("source_discovery_run"),
        "step": step,
        "analysis_arm": arm,
        "top_k": top_k,
        "candidates": promoted_md,
        "note": (
            "Promoted candidates for explicit preparation, PROPKA audit, and bounded GPU MD. "
            "No GPU compute has been launched; cofold geometry does not establish activity."
        ),
    }
    atomic_write_json(md_manifest_path, md_manifest, allow_nan=False)

    design_handoff_path = dest / f"design_handoff_step{step}_{arm}.json"
    design_handoff = {
        "kind": "enzyme_design_request",
        "purpose": "enzyme_design",
        "recipient": "enzyme_design_agent",
        "requested_tool": None,
        "source_report": {"path": str(report_path), "sha256": file_digest(report_path)},
        "source_discovery_run": report.get("source_discovery_run"),
        "target": report.get("target"),
        "reference_candidates": [
            {
                "accession": c["accession"],
                "enzyme_name": c["enzyme_name"],
                "family_id": c["family_id"],
                "weighted_attack_angle_deg": c["weighted_attack_angle_deg"],
                "harmonic_distance_A": c["harmonic_distance_A"],
                "near_attack_fraction": c["near_attack_fraction"],
                "holo_seating_status": c["holo_seating_status"],
                "selected_pose": c["selected_pose"],
            }
            for c in promoted_md
        ],
        "note": (
            "Strategy-open design handoff from family-cofold results. enzyme_design_agent "
            "should decide between saturation redesign and generative de novo strategies."
        ),
    }
    atomic_write_json(design_handoff_path, design_handoff, allow_nan=False)

    return {
        "status": "promoted",
        "n_candidates": len(promoted_md),
        "md_promotion_path": str(md_manifest_path),
        "design_handoff_path": str(design_handoff_path),
        "design_command": [
            "python",
            "-m",
            "enzyme_design_agent",
            "--handoff",
            str(design_handoff_path),
        ],
        "candidates": [c["accession"] for c in promoted_md],
    }


def build_design_validation_task(
    design_run_dir: str | Path,
    *,
    target: dict[str, Any],
    sequences: dict[str, str] | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a standard ModelingAgent validation task from an enzyme_design_agent run.

    Closes the Design -> Structural Modeling validation loop so designed variants
    are evaluated under the exact same frozen Boltz-2 / geometry contract as
    natural enzymes from discovery.
    """
    run_dir = Path(design_run_dir).expanduser().resolve(strict=True)
    manifest_path = run_dir / "manifest.json"
    handoff_path = run_dir / "design_output_handoff.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing design-agent manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("agent") != "enzyme_design_agent":
        raise ValueError(f"Expected enzyme_design_agent manifest at {manifest_path}")

    output_handoff = json.loads(handoff_path.read_text()) if handoff_path.is_file() else {}

    collected: dict[str, str] = {}
    if sequences:
        for cid, seq in sequences.items():
            cleaned = "".join(str(seq).split()).upper()
            if cleaned:
                collected[str(cid)] = cleaned

    # Also auto-extract any explicit {id, sequence} entries recorded in saved design artifacts.
    artifacts_dir = run_dir / "artifacts" / "design"
    if artifacts_dir.is_dir():
        for art in sorted(artifacts_dir.glob("*.json")):
            try:
                doc = json.loads(art.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            data = (doc.get("result") or {}).get("data")
            if not isinstance(data, dict):
                continue
            for key in ("designs", "variants", "candidates"):
                rows = data.get(key)
                if not isinstance(rows, list):
                    continue
                for idx, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    seq = row.get("sequence")
                    if isinstance(seq, str) and seq.strip():
                        cid = str(row.get("id") or row.get("name") or f"{art.stem}_{idx}")
                        collected.setdefault(cid, "".join(seq.split()).upper())

    if not collected:
        raise ValueError(
            "No designed protein sequences found in design run artifacts; "
            "pass sequences={design_id: aa_sequence} explicitly."
        )

    candidates = []
    for cid, seq in sorted(collected.items()):
        candidates.append(
            {
                "accession": cid,
                "sequence": seq,
                "sequence_sha256": hashlib.sha256(seq.encode("ascii")).hexdigest(),
                "evidence": {
                    "origin": "enzyme_design_agent",
                    "design_run_dir": str(run_dir),
                    "design_strategy": manifest.get("strategy"),
                    "design_manifest": {"path": str(manifest_path), "sha256": file_digest(manifest_path)},
                    "design_output_handoff": (
                        {"path": str(handoff_path), "sha256": file_digest(handoff_path)}
                        if handoff_path.is_file()
                        else None
                    ),
                },
            }
        )

    task = {
        "schema_version": "design-validation-task-1.0",
        "kind": "design_validation_task",
        "target": target,
        "source_design_run": str(run_dir),
        "design_strategy": manifest.get("strategy"),
        "source_handoff": output_handoff.get("source_handoff") or manifest.get("source_handoff"),
        "candidates": candidates,
        "note": (
            "Designed sequences routed into structural_modeling_agent for independent "
            "Boltz-2 cofold and frozen geometry validation; generation alone does not establish activity."
        ),
    }

    dest = Path(out_path).resolve() if out_path else run_dir / "structural_validation_task.json"
    atomic_write_json(dest, task, allow_nan=False)
    return {
        "status": "prepared",
        "task_path": str(dest),
        "task_sha256": file_digest(dest),
        "n_candidates": len(candidates),
        "candidate_ids": [c["accession"] for c in candidates],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prom = sub.add_parser("promote-cofold", help="Promote top-K family-cofold candidates to MD & Design")
    p_prom.add_argument("--family-dir", required=True, type=Path)
    p_prom.add_argument("--top-k", type=int, default=3)
    p_prom.add_argument("--step", type=int, default=1)
    p_prom.add_argument("--arm", default="unrestrained")
    p_prom.add_argument("--md-spec", type=Path, default=None)
    p_prom.add_argument("--ph", type=float, default=8.0)
    p_prom.add_argument("--prep-python", default=DEFAULT_PREP_PYTHON)
    p_prom.add_argument("--md-python", default=DEFAULT_MD_PYTHON)
    p_prom.add_argument("--out-dir", type=Path, default=None)

    p_val = sub.add_parser("validate-design", help="Build a ModelingAgent validation task from a design run")
    p_val.add_argument("--design-run", required=True, type=Path)
    p_val.add_argument("--target-json", required=True, type=Path)
    p_val.add_argument("--sequences-json", type=Path, default=None)
    p_val.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "promote-cofold":
        res = promote_family_cofold(
            args.family_dir,
            top_k=args.top_k,
            step=args.step,
            arm=args.arm,
            md_spec=args.md_spec,
            ph=args.ph,
            prep_python=args.prep_python,
            md_python=args.md_python,
            out_dir=args.out_dir,
        )
    else:
        target = json.loads(args.target_json.read_text())
        seqs = json.loads(args.sequences_json.read_text()) if args.sequences_json else None
        res = build_design_validation_task(
            args.design_run,
            target=target,
            sequences=seqs,
            out_path=args.out,
        )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
