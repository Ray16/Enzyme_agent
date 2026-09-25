#!/usr/bin/env python3
"""Generic tests for the handoff resolution compiler.

Deliberately domain-agnostic: no specific enzyme/target is baked in. Fixtures use
a generic carboxylate ligand and synthetic candidates whose mechanism (activation
mode) and oligomeric state are varied to exercise family-based metal reasoning.

Run under the base env (RDKit):
  CONDA_NO_PLUGINS=true ~/miniforge3/bin/python -m unittest discover \
    -s modeling_reasoning/tests -p 'test_*.py' -v
"""
import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

import resolve_handoff as rh  # noqa: E402
from literature import LiteratureKnowledge  # noqa: E402

GENERIC_MANIFEST = {
    "target_id": "generic-test",
    "task_type": "cofolding",
    "step": {"index": 1, "chosen_because": "activation half-reaction"},
    "scoring_policy": "unrestrained",
    "features": {"non_protein_participant": True, "task_has_geometric_success_criteria": True},
    "resolutions": {
        "enzyme_state": {
            "chains": [{"source": "full_sequence"}],
            "copies": 1,
            "state": "intact",
            "origin": "source",
            "decision_point": "participants",
            "rationale": "intact single-chain enzyme",
        },
        "substrates": [
            {
                "match": {"component_index": 1},
                "include_in_model": True,
                "modeling_smiles": "CC(=O)[O-]",
                "role": "carboxylate_substrate",
                "affinity_target": True,
                "review_status": "reviewed",
                "origin": "experience",
                "decision_point": "substrate-representation",
                "rationale": "modeled as the anion",
            },
            {
                "match": {"component_index": 2},
                "include_in_model": False,
                "origin": "source",
                "decision_point": "participants",
                "rationale": "co-substrate omitted for this half-reaction",
            },
        ],
        "catalytic_metal": {
            "strategy": "family_reason",
            "origin": "database",
            "decision_point": "participants",
            "rationale": "per-active-site metal reasoned from mechanism, scaled by oligomeric state",
        },
        "geometry": {
            "origin": "inference",
            "decision_point": "pose-or-conformer-selection-criterion",
            "rationale": "near-attack geometry derived from the declared step",
        },
    },
}

# Step-2 manifest: acyl-adenylate intermediate + amine; ATP is consumed and absent.
GENERIC_STEP2_MANIFEST = {
    "target_id": "generic-test-step2",
    "task_type": "cofolding",
    "step": {"index": 2, "chosen_because": "aminolysis half-reaction"},
    "scoring_policy": "unrestrained",
    "features": {"non_protein_participant": True, "task_has_geometric_success_criteria": True},
    "resolutions": {
        "enzyme_state": {
            "chains": [{"source": "full_sequence"}], "copies": 1, "state": "intact",
            "origin": "source", "decision_point": "participants", "rationale": "intact enzyme",
        },
        "substrates": [
            {
                "match": {"component_index": 1},
                "include_in_model": True,
                "modeling_smiles": "CC(=O)OP(=O)(O)O",
                "role": "acyl_adenylate_intermediate",
                "affinity_target": True,
                "review_status": "reviewed",
                "origin": "experience",
                "decision_point": "substrate-representation",
                "rationale": "pre-formed acyl-adenylate intermediate",
            },
            {
                "match": {"component_index": 2},
                "include_in_model": True,
                "modeling_smiles": "CCN",
                "role": "amine_substrate",
                "review_status": "reviewed",
                "origin": "experience",
                "decision_point": "participants",
                "rationale": "attacking amine",
            },
        ],
        "catalytic_metal": {
            "strategy": "family_reason", "origin": "database",
            "decision_point": "participants", "rationale": "family per-site metal",
        },
        "geometry": {
            "origin": "inference", "decision_point": "pose-or-conformer-selection-criterion",
            "rationale": "amine N -> acyl carbonyl C, from the reaction coordinate",
        },
    },
}


def _candidate(accession, activation_mode="acyl_adenylate"):
    reactive = {"acyl_adenylate": ("PA", "O3A"), "acyl_phosphate": ("PG", "O3B")}[activation_mode]
    substrate = lambda idx, smiles: {  # noqa: E731
        "component_index": idx,
        "source_smiles": smiles,
        "source_representation": {"valid": True, "canonical_isomeric_smiles": smiles},
        "modeling_smiles": smiles,
        "modeling_representation": {"valid": True, "stereochemistry": {"status": "not_applicable"}},
        "role": None,
        "include_in_model": True,
        "affinity_target": False,
        "review_status": "not_required",
    }
    return {
        "accession": accession,
        "evidence": {"identity": {"accession": accession}, "cofactor_annotations": {"names": []}},
        "modeling_input": {
            "schema_version": "2.0",
            "status": "unresolved",
            "method": "cofold",
            "enzyme_state": {"state": "specified", "chains": None},
            "target_substrates": [substrate(1, "CC(=O)O"), substrate(2, "NCCN")],
            "activation": {
                "status": "assigned",
                "activation_mode": activation_mode,
                "nucleotide": "ATP",
                "reactive_atom": reactive[0],
                "leaving_group_atom": reactive[1],
            },
            "cofactors": [{"name": "ATP", "ccd": "ATP", "count": 1, "role": "nucleotide_substrate"}],
            "geometry": None,
            "geometry_status": "unresolved",
            "unresolved": ["catalytically relevant enzyme state is unresolved"],
        },
        "selection": {"source": "evaluated_discovery_candidate"},
    }


def _handoff(*candidates):
    return {"schema_version": "2.0", "candidates": list(candidates)}


# Offline default: no UniProt record -> cross-check reference is a generic default.
_offline = lambda *a, **k: None  # noqa: E731
# A deliberately WRONG UniProt tally, to prove it does not drive placement.
def _fetch_says_four(url, timeout=30):
    return {"comments": [{"commentType": "COFACTOR", "cofactors": [{"name": "Mg(2+)"}],
                          "note": {"texts": [{"value": "Binds 4 Mg(2+) ions per subunit."}]}}]}


class ResolveHandoffTest(unittest.TestCase):
    def setUp(self):
        self.knowledge = LiteratureKnowledge.load()

    def _resolve(self, manifest, *candidates, fetch=_offline, step=None):
        handoff = _handoff(*[copy.deepcopy(c) for c in candidates])
        reports = rh.resolve_handoff(handoff, manifest, self.knowledge, step=step, fetch=fetch)
        return handoff, reports

    def test_resolved_is_renderer_ready(self):
        handoff, reports = self._resolve(GENERIC_MANIFEST, _candidate("X1"))
        mi = handoff["candidates"][0]["modeling_input"]
        self.assertEqual(mi["status"], "ready", reports["X1"]["errors"])
        self.assertEqual(rh.readiness_errors(mi), [])
        self.assertEqual(rh.execution_shape_errors(mi), [])

    def test_metal_reasoned_from_family(self):
        handoff, _ = self._resolve(
            GENERIC_MANIFEST,
            _candidate("ADENYLATE", "acyl_adenylate"),
            _candidate("PHOSPHATE", "acyl_phosphate"),
        )
        counts = {}
        for cand in handoff["candidates"]:
            mg = next(c for c in cand["modeling_input"]["cofactors"] if c["ccd"] == "MG")
            counts[cand["accession"]] = mg["count"]
        self.assertEqual(counts["ADENYLATE"], 1)  # single catalytic Mg per site
        self.assertEqual(counts["PHOSPHATE"], 2)  # two-metal mechanism

    def test_metal_scales_with_oligomeric_state(self):
        manifest = copy.deepcopy(GENERIC_MANIFEST)
        manifest["resolutions"]["enzyme_state"]["copies"] = 4  # tetramer -> 4 active sites
        handoff, _ = self._resolve(manifest, _candidate("TET", "acyl_adenylate"))
        mg = next(c for c in handoff["candidates"][0]["modeling_input"]["cofactors"] if c["ccd"] == "MG")
        self.assertEqual(mg["count"], 4)  # per_site(1) x active_sites(4)
        self.assertEqual(mg["basis"]["per_site_metal_count"], 1)
        self.assertEqual(mg["basis"]["active_sites"], 4)

    def test_uniprot_tally_is_reference_not_placement(self):
        # UniProt says "Binds 4"; family reasoning says 1 per site, 1 site -> place 1, not 4.
        handoff, _ = self._resolve(GENERIC_MANIFEST, _candidate("REF", "acyl_adenylate"), fetch=_fetch_says_four)
        mg = next(c for c in handoff["candidates"][0]["modeling_input"]["cofactors"] if c["ccd"] == "MG")
        self.assertEqual(mg["count"], 1)
        ref = mg["basis"]["uniprot_reference"]
        self.assertEqual(ref["uniprot_count"], 4)
        self.assertFalse(ref["agrees_with_per_site"])

    def test_ligand_reverified(self):
        handoff, _ = self._resolve(GENERIC_MANIFEST, _candidate("X1"))
        subs = handoff["candidates"][0]["modeling_input"]["target_substrates"]
        acceptor = next(s for s in subs if s["component_index"] == 1)
        self.assertEqual(acceptor["modeling_representation"]["formal_charge"], -1)
        self.assertTrue(acceptor["affinity_target"])

    def test_cosubstrate_excluded(self):
        handoff, _ = self._resolve(GENERIC_MANIFEST, _candidate("X1"))
        subs = handoff["candidates"][0]["modeling_input"]["target_substrates"]
        self.assertFalse(next(s for s in subs if s["component_index"] == 2).get("include_in_model"))

    def test_no_restraints_but_geometry_recorded(self):
        handoff, _ = self._resolve(GENERIC_MANIFEST, _candidate("X1"))
        mi = handoff["candidates"][0]["modeling_input"]
        self.assertEqual(mi["scoring_policy"]["restraints"], "unrestrained")
        self.assertTrue(mi["geometry"]["measurement_only"])
        self.assertNotIn("constraints", mi)

    def test_literature_provenance_attached(self):
        handoff, _ = self._resolve(GENERIC_MANIFEST, _candidate("X1"))
        mi = handoff["candidates"][0]["modeling_input"]
        basis = mi["resolution_provenance"]["substrates"][0]["literature_basis"]
        self.assertTrue(basis, "substrate resolution should cite PDF-extracted rules")
        papers = {p for block in basis for p in block["papers"]}
        self.assertTrue(any(p.startswith("PMID:") for p in papers))

    # --- step generality ---

    def test_step_must_be_declared(self):
        manifest = copy.deepcopy(GENERIC_MANIFEST)
        manifest.pop("step")
        handoff, reports = self._resolve(manifest, _candidate("X1"))  # no --step either
        mi = handoff["candidates"][0]["modeling_input"]
        self.assertEqual(mi["status"], "needs_review")
        self.assertTrue(any("must be declared" in e for e in mi["unresolved"]))

    def test_step1_geometry_is_activation_attack(self):
        handoff, _ = self._resolve(GENERIC_MANIFEST, _candidate("X1", "acyl_adenylate"))
        g = handoff["candidates"][0]["modeling_input"]["geometry"]
        self.assertEqual(g["step"], 1)
        self.assertEqual(g["nucleophile"]["role"], "carboxylate_substrate")
        self.assertEqual(g["electrophile"]["role"], "NTP")
        self.assertEqual(g["electrophile"]["atom"], "PA")  # alpha-phosphate of ATP

    def test_step2_is_correct_and_ready(self):
        handoff, reports = self._resolve(GENERIC_STEP2_MANIFEST, _candidate("X1", "acyl_adenylate"))
        mi = handoff["candidates"][0]["modeling_input"]
        self.assertEqual(mi["status"], "ready", reports["X1"]["errors"])
        # ATP consumed in step 1 -> absent in step 2
        self.assertNotIn("ATP", [c["ccd"] for c in mi["cofactors"]])
        # geometry is the amine attack, not the phosphate attack
        g = mi["geometry"]
        self.assertEqual(g["step"], 2)
        self.assertEqual(g["nucleophile"]["role"], "amine_substrate")
        self.assertEqual(g["electrophile"]["role"], "acyl_adenylate_intermediate")

    def test_wrong_step_is_blocked(self):
        # Step-1 participants (carboxylate + ATP) but the agent declares step 2:
        # the fail-closed gate must refuse to model it.
        handoff, _ = self._resolve(GENERIC_MANIFEST, _candidate("X1", "acyl_adenylate"), step=2)
        mi = handoff["candidates"][0]["modeling_input"]
        self.assertEqual(mi["status"], "needs_review")
        joined = " ".join(mi["unresolved"])
        self.assertIn("acyl_adenylate_intermediate", joined)  # required but not modeled
        self.assertIn("amine_substrate", joined)


if __name__ == "__main__":
    unittest.main()
