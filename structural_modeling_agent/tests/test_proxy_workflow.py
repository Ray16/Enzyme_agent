"""Coordinate invariance, chemistry declarations, and shared descriptor coverage."""
from dataclasses import replace
import numpy as np
import pytest

from structural_modeling_agent.proxies import Atom, Pose, ReactiveSite, build_ensemble, evaluate
from structural_modeling_agent.proxies.workflow import evaluate_plan
from structural_modeling_agent.proxies.calibration import screen, partial_spearman


def fixture():
    ca = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
    protein = np.array([[0.,0.,2.], [1.,0.,2.]])
    atoms = (Atom("A", "10", "NZ", "LYS", "N", protein[0], True),
             Atom("A", "11", "N", "ALA", "N", protein[1], True),
             Atom("E", "1", "C1", "LIG", "C", np.array([0.,0.,3.]), False))
    pose = Pose(ca, protein, protein, protein, {"E": [("C1", "C", atoms[-1].xyz)]}, atoms=atoms,
                confidence={"complex_plddt": .9, "iptm": .7, "ligand_iptm": .6,
                            "pair_chains_iptm": {"0": {"3": .95}, "3": {"0": .35}}})
    site = ReactiveSite(np.array([0.,0.,3.]), np.zeros(3), leaving=np.array([0.,0.,-1.]),
                        charge_centre=np.zeros(3), angle_reference=np.array([0.,0.,-1.]))
    return pose, site


def plan(ids, context=None):
    return {"schema_version": "1.0", "mechanism": "declared mechanism",
            "reaction_step": "declared step", "assay_endpoint": "matched product measurement",
            "context": context or {}, "hypotheses": [{"proxy_id": pid,
                "rationale": "Mechanistic hypothesis declared before labels", "expected_direction": "+",
                "evidence_ids": ["inference:reviewed"]} for pid in ids]}


def test_rigid_motion_preserves_contact_and_environment_proxies():
    pose, site = fixture()
    rot = np.array([[0.,-1.,0.], [1.,0.,0.], [0.,0.,1.]])
    shift = np.array([100.,30.,-50.])
    mv = lambda x: rot @ x + shift
    other = replace(pose, ca=pose.ca @ rot.T+shift, protein=pose.protein @ rot.T+shift,
                    donors=pose.donors @ rot.T+shift, bases=pose.bases @ rot.T+shift,
                    atoms=tuple(replace(a, xyz=mv(a.xyz)) for a in pose.atoms),
                    ligands={"E": [("C1", "C", mv(pose.ligands["E"][0][2]))]})
    changed = replace(site, nucleophile=mv(site.nucleophile), electrophile=mv(site.electrophile),
                      leaving=mv(site.leaving), charge_centre=mv(site.charge_centre), angle_reference=mv(site.angle_reference))
    single = evaluate(build_ensemble([pose], [site]))
    multiple = evaluate(build_ensemble([pose, other]*3, [site, changed]*3))
    for pid in ("nucleophile_burial", "ts_charge_stabilisation", "base_proximity", "leaving_group_stabilisation"):
        assert multiple[pid] == pytest.approx(single[pid])
    assert multiple["reactive_preorganisation"] == pytest.approx(0., abs=1e-10)


def test_alignment_failure_does_not_fabricate_positional_consistency():
    pose, site = fixture()
    bad = replace(pose, ca=np.array([[0.,0.,0.]]))
    result = evaluate(build_ensemble([pose, bad]*3, [site]*6))
    assert result["reactive_preorganisation"] is None
    assert result["approach_consistency"] is None
    assert result["approach_distance"] == pytest.approx(3.)
    with pytest.raises(ValueError):
        build_ensemble([pose], [])


def test_explicit_angle_reference_can_differ_from_leaving_atom():
    pose, site = fixture()
    reference = np.array([np.sin(np.deg2rad(107.)), 0., np.cos(np.deg2rad(107.))])
    site = replace(site, angle_reference=reference, ideal_angle=107.)
    result = evaluate(build_ensemble([pose]*6, [site]*6))
    assert result["trajectory_angle_deviation"] == pytest.approx(0., abs=.01)
    assert result["near_attack_fraction"] == 1.
    missing = replace(site, angle_reference=None, leaving=None)
    assert evaluate(build_ensemble([pose]*6, [missing]*6))["near_attack_fraction"] is None


def test_pair_confidence_requires_an_explicit_entity_map_and_preserves_direction():
    pose, site = fixture()
    ids = ["protein_substrate_iptm", "substrate_protein_iptm"]
    result = evaluate_plan([pose], [site], plan(ids))
    assert all(v["status"] == "unavailable" for v in result["measurements"].values())
    context = {"confidence_chain_map": {"A": "0", "E": "3"},
               "confidence_pairs": [{"protein_chain": "A", "substrate_chain": "E"}]}
    result = evaluate_plan([pose], [site], plan(ids, context))["measurements"]
    assert result[ids[0]]["value"] == .95
    assert result[ids[1]]["value"] == .35


def test_shared_descriptors_cover_accommodation_environment_and_consistency():
    pose, site = fixture()
    selector = {"chain": "A", "residue": "10", "atom": "NZ"}
    context = {"substrate_chains": ["E"], "catalytic_base_atoms": [selector],
               "catalytic_donor_atoms": [selector], "charges": [{**selector, "charge": 1.}],
               "charge_model": "declared point charge", "charge_provenance": "test fixture",
               "unlisted_atom_policy": "zero", "potential_target": "electrophile",
               "field_axis": ["nucleophile", "electrophile"]}
    ids = ["ligand_contact_fraction", "annotated_base_distance", "annotated_donor_contacts",
           "reactive_site_potential", "projected_electric_field", "electric_field_variability",
           "contact_consistency", "ligand_positional_dispersion"]
    result = evaluate_plan([pose]*6, [site]*6, plan(ids, context))["measurements"]
    assert all(v["status"] == "computed" for v in result.values())
    assert result["ligand_contact_fraction"]["value"] == 1.
    assert result["annotated_base_distance"]["value"] == 1.
    assert result["reactive_site_potential"]["value"] == .5
    assert result["projected_electric_field"]["value"] == .25
    assert result["electric_field_variability"]["value"] == 0.
    assert result["contact_consistency"]["value"] == 1.
    assert result["ligand_positional_dispersion"]["value"] == pytest.approx(0., abs=1e-10)


def test_sasa_matches_an_isolated_sphere_and_records_defaults():
    pose, site = fixture()
    atom = replace(pose.atoms[-1], xyz=np.array([100.,100.,100.]))
    pose = replace(pose, atoms=(*pose.atoms[:-1], atom))
    p = plan(["reactive_group_sasa"], {"reactive_group_atoms": [{"chain": "E", "residue": "1", "atom": "C1"}]})
    result = evaluate_plan([pose], [site], p)["measurements"]["reactive_group_sasa"]
    assert result["value"] == pytest.approx(4*np.pi*3.1**2)
    assert result["parameters"] == {"probe_radius_A": 1.4, "sphere_points": 96}


def test_empty_contacts_are_not_reported_as_conserved_binding():
    pose, site = fixture()
    pose = replace(pose, atoms=(*pose.atoms[:-1], replace(pose.atoms[-1], xyz=np.ones(3)*100)))
    result = evaluate_plan([pose]*2, [site]*2, plan(["contact_consistency"], {"substrate_chains": ["E"]}))
    assert result["measurements"]["contact_consistency"]["value"] == 0.


def test_unused_parameters_and_targeted_followups_are_explicit():
    pose, site = fixture()
    p = plan(["complex_plddt"])
    p["hypotheses"][0]["parameters"] = {"distance_gate_A": 3.5}
    with pytest.raises(ValueError, match="unsupported parameters"):
        evaluate_plan([pose], [site], p)
    result = evaluate_plan([pose], [site], plan(["access_channel_geometry"]))
    assert result["measurements"]["access_channel_geometry"]["status"] == "requires_targeted_extractor"


def test_confounded_feature_is_not_promoted_despite_high_raw_auroc():
    z = np.arange(30., dtype=float)
    proxy = type("Proxy", (), {"id": "candidate", "expected_direction": "+"})()
    verdict = screen({"candidate": z}, z, [proxy], confound=z, n_perm=40)[0]
    assert verdict.score == 1.
    assert not verdict.promoted and not verdict.confound_ok
    assert np.isnan(partial_spearman(z, z, z))
    missing = screen({"candidate": z}, z, [proxy], n_perm=40)[0]
    assert not missing.promoted and "not evaluated" in missing.note
