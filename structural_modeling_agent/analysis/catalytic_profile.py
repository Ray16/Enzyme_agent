"""Unified, non-collapsed multi-axis catalytic profile and failure-mode synthesizer.

Combines independent structural, dynamic, energetic, and developability evidence
into a single auditable profile without collapsing distinct physical axes into an
arbitrary composite score. Missing axes remain explicitly ``not_evaluated``.
"""
from __future__ import annotations

from typing import Any


AXIS_NAMES = (
    "mechanism_gate",
    "pocket_envelope",
    "access",
    "reactive_persistence",
    "ts_and_preorganization",
    "side_reactions",
    "barrier_status",
    "developability",
)


def _axis_summary(data: dict[str, Any] | None, *, default_status: str = "not_evaluated") -> dict[str, Any]:
    if not data:
        return {"status": default_status, "evidence": None}
    status = data.get("status") or data.get("verdict") or "evaluated"
    return {"status": status, "evidence": data}


def build_catalytic_profile(
    candidate_id: str,
    *,
    mechanism_gate: dict[str, Any] | None = None,
    pocket_envelope: dict[str, Any] | None = None,
    access: dict[str, Any] | None = None,
    reactive_persistence: dict[str, Any] | None = None,
    ts_and_preorganization: dict[str, Any] | None = None,
    side_reactions: dict[str, Any] | None = None,
    barrier_status: dict[str, Any] | None = None,
    developability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a non-collapsed multi-axis CatalyticProfile with failure-mode diagnosis.

    Separates:
      * ``hard_reject``: mechanistic incompatibility, developability veto, buried
        steric exclusion (native passes, target clashes), or severe competing side
        reaction that outcompetes the desired pathway.
      * ``soft_evolvable_defect``: loose/drifting pocket, distal-tail mismatch,
        narrow gating loop, or suboptimal preorganization — actionable targets
        for 2-6 mutations in enzyme_design_agent.
      * ``modeling_artifact``: native control fails alongside target (fix setup
        before judging the scaffold).
    """
    axes = {
        "mechanism_gate": _axis_summary(mechanism_gate),
        "pocket_envelope": _axis_summary(pocket_envelope),
        "access": _axis_summary(access),
        "reactive_persistence": _axis_summary(reactive_persistence),
        "ts_and_preorganization": _axis_summary(ts_and_preorganization),
        "side_reactions": _axis_summary(side_reactions),
        "barrier_status": _axis_summary(barrier_status),
        "developability": _axis_summary(developability),
    }

    failure_modes: list[dict[str, Any]] = []
    design_targets: list[dict[str, Any]] = []

    # 1. Mechanism / setup hard gate
    if mechanism_gate:
        m_status = mechanism_gate.get("status")
        if m_status == "fail":
            failure_modes.append({
                "axis": "mechanism_gate",
                "code": "mechanism_or_setup_incompatible",
                "severity": "hard_reject",
                "detail": mechanism_gate.get("reason") or "Setup or mechanism validation failed.",
            })
        elif m_status == "abstain":
            failure_modes.append({
                "axis": "mechanism_gate",
                "code": "mechanism_unresolved",
                "severity": "warning",
                "detail": mechanism_gate.get("reason") or "Required setup evidence is incomplete.",
            })

    # 2. Developability hard veto vs flag
    if developability:
        verdict = developability.get("verdict")
        if verdict == "veto":
            failure_modes.append({
                "axis": "developability",
                "code": "developability_veto",
                "severity": "hard_reject",
                "detail": "; ".join(developability.get("reasons") or ["Curated developability veto."]),
                "exploration_trigger": developability.get("exploration_trigger"),
            })
        elif verdict == "flag":
            failure_modes.append({
                "axis": "developability",
                "code": "developability_flag",
                "severity": "warning",
                "detail": "; ".join(developability.get("reasons") or ["Predicted developability below floor."]),
                "exploration_trigger": developability.get("exploration_trigger"),
            })

    # 3. Pocket envelope & native-vs-target diagnosis
    if pocket_envelope:
        diag = pocket_envelope.get("diagnosis")
        if diag == "modeling_setup_issue":
            failure_modes.append({
                "axis": "pocket_envelope",
                "code": "native_control_also_fails",
                "severity": "modeling_artifact",
                "detail": "Both native control and target exhibit clashes or poor seating; verify assembly, protonation, and cofactors before scoring.",
            })
        elif diag == "buried_steric_exclusion":
            failure_modes.append({
                "axis": "pocket_envelope",
                "code": "buried_steric_exclusion",
                "severity": "hard_reject",
                "detail": "Target clashes inside a buried pocket where the native substrate fits cleanly.",
            })
        elif diag == "pocket_too_loose_slack":
            failure_modes.append({
                "axis": "pocket_envelope",
                "code": "pocket_too_loose_drifter",
                "severity": "soft_evolvable_defect",
                "detail": "Target is smaller than the native envelope; cavity-shrinking mutations can improve positioning.",
            })
        elif diag == "distal_tail_buried_clash":
            clashing = pocket_envelope.get("clashing_residues") or []
            failure_modes.append({
                "axis": "pocket_envelope",
                "code": "distal_tail_clash",
                "severity": "soft_evolvable_defect",
                "detail": "Distal (non-reacting) tail clashes with buried wall residues.",
                "residues": clashing,
            })
            if clashing:
                design_targets.append({
                    "defect": "distal_tail_clash",
                    "residues": clashing,
                    "suggested_strategy": "Relieve steric/electrostatic clash at distal wall residues.",
                })

    # 4. Access / channel topology
    if access:
        site = access.get("site") or {}
        burial = (site.get("burial_fraction") or {}).get("mean")
        n_ch = access.get("n_channels")
        if burial is not None and burial >= 0.85 and n_ch == 0 and not site.get("surface_exposed"):
            failure_modes.append({
                "axis": "access",
                "code": "buried_no_static_channel",
                "severity": "warning",
                "detail": "Deeply buried site with no open static tunnel; check gating-loop flexibility or MD breathing.",
            })
        channels = access.get("channels") or []
        if channels:
            lining = [r["residue"] for r in (channels[0].get("bottleneck_lining_residues") or [])[:5]]
            bn = (channels[0].get("bottleneck_A") or {}).get("mean")
            if bn is not None and bn < 1.2 and lining:
                design_targets.append({
                    "defect": "narrow_channel_bottleneck",
                    "residues": lining,
                    "suggested_strategy": "Candidate bottleneck-lining residues if transport proves rate-limiting.",
                })

    # 5. Reactive persistence (Boltz prior vs MD persistence)
    if reactive_persistence:
        md_nac = reactive_persistence.get("md_nac_fraction")
        boltz_nac = reactive_persistence.get("boltz_nac_fraction")
        if md_nac is not None and md_nac < 0.05:
            if boltz_nac is not None and boltz_nac >= 0.2:
                failure_modes.append({
                    "axis": "reactive_persistence",
                    "code": "cofold_poised_md_drifter",
                    "severity": "soft_evolvable_defect",
                    "detail": f"Poised in cofold (NAC={boltz_nac:.2f}) but drifts in MD (NAC={md_nac:.2f}); strengthen substrate anchoring.",
                })
            else:
                failure_modes.append({
                    "axis": "reactive_persistence",
                    "code": "low_md_reactive_persistence",
                    "severity": "soft_evolvable_defect",
                    "detail": f"Low MD near-attack occupancy (NAC={md_nac:.2f}).",
                })

    # 6. Preorganization & TS/intermediate stabilization
    if ts_and_preorganization:
        reorg_rmsd = (ts_and_preorganization.get("apo_holo_reorganization") or {}).get("catalytic_heavy_rmsd_A")
        if reorg_rmsd is not None and reorg_rmsd > 2.5:
            failure_modes.append({
                "axis": "ts_and_preorganization",
                "code": "high_reorganization_penalty",
                "severity": "warning",
                "detail": f"Catalytic constellation shifts by {reorg_rmsd:.2f} A between apo and holo states.",
            })
        field = ts_and_preorganization.get("electric_field") or {}
        e_mean = field.get("projected_field_MV_per_cm_mean")
        expected_sign = field.get("expected_favorable_sign")
        if e_mean is not None and expected_sign in (-1, 1) and (e_mean * expected_sign) < 0:
            failure_modes.append({
                "axis": "ts_and_preorganization",
                "code": "opposing_electrostatic_field",
                "severity": "soft_evolvable_defect",
                "detail": f"Mean projected electric field ({e_mean:.1f} MV/cm) opposes the declared reaction dipole.",
            })

    # 7. Competing side reactions
    if side_reactions:
        for pathway in side_reactions.get("pathways", []):
            occ = pathway.get("occupancy_fraction")
            warn_cut = pathway.get("warning_threshold", 0.30)
            if occ is not None and occ >= warn_cut:
                failure_modes.append({
                    "axis": "side_reactions",
                    "code": f"competing_{pathway['name']}",
                    "severity": "soft_evolvable_defect" if occ < 0.75 else "hard_reject",
                    "detail": f"Competing pathway '{pathway['name']}' reaches near-attack geometry in {occ:.1%} of evaluated frames.",
                })

    # 8. Barrier status
    if barrier_status and barrier_status.get("status") == "fail":
        failure_modes.append({
            "axis": "barrier_status",
            "code": "barrier_gate_failed",
            "severity": "warning",
            "detail": barrier_status.get("reason") or "Barrier calculation did not pass strict reporting gate.",
        })

    # Derive overall evolvability class without collapsing numerical scores
    severities = {fm["severity"] for fm in failure_modes}
    evaluated_count = sum(1 for a in axes.values() if a["status"] != "not_evaluated")
    if "modeling_artifact" in severities:
        evolvability_class = "needs_setup_review"
    elif "hard_reject" in severities:
        evolvability_class = "hard_reject"
    elif "soft_evolvable_defect" in severities:
        evolvability_class = "evolvable_with_mutations"
    elif evaluated_count == 0:
        evolvability_class = "incomplete"
    else:
        evolvability_class = "promising_scaffold"

    return {
        "schema_version": "catalytic-profile-1.0",
        "candidate_id": candidate_id,
        "evolvability_class": evolvability_class,
        "axes_evaluated": [k for k, v in axes.items() if v["status"] != "not_evaluated"],
        "axes_missing": [k for k, v in axes.items() if v["status"] == "not_evaluated"],
        "axes": axes,
        "failure_modes": failure_modes,
        "design_targets": design_targets,
        "scope": "Multi-axis structural/computational profile; distinct from experimentally demonstrated enzyme activity.",
    }
