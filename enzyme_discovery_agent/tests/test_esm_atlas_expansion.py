"""Tests for the ESM Metagenomic Atlas expansion tool (offline; network stubbed).

The AlphaFold seed fetch and the Foldseek submit/poll/result HTTP calls are stubbed,
so these exercise the tool's parsing + registration logic deterministically. Atlas
hits are metagenomic MGYP identifiers (not UniProt), carry a sequence, and no
annotation -- the registration path must accept them and store the sequence.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    load_candidate_registry,
    register_atlas_homologs,
    update_candidate_registry,
)
from enzyme_discovery_agent.gates.ec_triage import auto_screen_candidates, _auto_evaluate_all
from enzyme_discovery_agent.tools.reasoning import evaluate_candidate, screen_candidates
from enzyme_discovery_agent.tools import esm_atlas_expansion as ax


_PAYLOAD = {"results": [{"db": "mgnify_esm30", "alignments": [[
    {"target": "MGYP000932894257.pdb.gz", "prob": 1.0, "seqId": 59.6,
     "alnLength": 360, "tSeq": "MSEQONEAAAA"},
    {"target": "MGYP000386953019.pdb.gz", "prob": 0.8, "seqId": 32.8,
     "alnLength": 370, "tSeq": "MSEQTWOBBBB"},
    {"target": "MGYP000932894257.pdb.gz", "prob": 0.4, "seqId": 20.0,
     "alnLength": 100, "tSeq": "DUP"},          # duplicate target -> deduped
    {"target": "6xyz_A", "prob": 0.9},           # non-MGnify -> skipped
]]}]}


class _Resp:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload


class TestParse(unittest.TestCase):
    def test_parse_dedupes_and_filters(self):
        homologs = ax._parse_atlas_results(_PAYLOAD, "SEED01")
        ids = [h["mgnify_id"] for h in homologs]
        self.assertEqual(ids, ["MGYP000932894257", "MGYP000386953019"])  # sorted by prob, deduped
        self.assertEqual(homologs[0]["probability"], 1.0)
        self.assertEqual(homologs[0]["sequence"], "MSEQONEAAAA")
        self.assertEqual(homologs[0]["source_database"], "ESMAtlas")


class TestTool(unittest.TestCase):
    def test_requires_accession(self):
        self.assertEqual(ax.search_esm_atlas_homologs("")["status"], "error")

    def test_full_flow_stubbed(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(ax, "_fetch_alphafold_structure", return_value="ATOM ..."), \
             mock.patch.object(ax.requests, "post", return_value=_Resp({"id": "TK1"})), \
             mock.patch.object(ax.requests, "get") as get, \
             mock.patch.object(ax.time, "sleep"):
            get.side_effect = [_Resp({"status": "COMPLETE"}), _Resp(_PAYLOAD)]
            update_candidate_registry(Path(d), "Q46E79", "evaluated")
            out = ax.search_esm_atlas_homologs("Q46E79", min_probability=0.5, run_dir=d)
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["data"]["homologs"][0]["mgnify_id"], "MGYP000932894257")

    def test_no_seed_structure_partial(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ax, "_fetch_alphafold_structure", return_value=None):
            update_candidate_registry(Path(d), "Q46E79", "evaluated")
            out = ax.search_esm_atlas_homologs("Q46E79", run_dir=d)
        self.assertEqual(out["status"], "partial")
        self.assertEqual(out["count"], 0)


class TestRegistration(unittest.TestCase):
    def test_registers_mgnify_ids_with_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            result = {"data": {"seed_accession": "Q46E79",
                               "search_method": "ESM Atlas Foldseek",
                               "homologs": [
                                   {"mgnify_id": "MGYP000932894257", "probability": 1.0,
                                    "sequence_identity": 59.6, "sequence": "MSEQ",
                                    "source_database": "ESMAtlas"},
                                   {"mgnify_id": "not_an_id"},  # rejected
                               ]}}
            registered = register_atlas_homologs(run_dir, result)
            self.assertEqual(registered, ["MGYP000932894257"])
            reg = load_candidate_registry(run_dir)
            cand = reg["candidates"]["MGYP000932894257"]
            self.assertEqual(current_candidate_status(cand), "discovered")
            self.assertEqual(cand["metadata"]["sequence"], "MSEQ")
            self.assertTrue(cand["metadata"]["is_metagenomic"])
            self.assertEqual(cand["metadata"]["structural_homology"]["seed_accession"], "Q46E79")

    def test_atlas_hit_can_be_screened_and_evaluated_without_uniprot(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            update = {
                "data": {
                    "seed_accession": "Q46E79",
                    "homologs": [{
                        "mgnify_id": "MGYP000932894257",
                        "probability": 0.95,
                        "sequence": "MSEQUENCE",
                    }],
                },
            }
            from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
            update_candidate_registry(
                run_dir, "Q46E79", "evaluated",
                metadata={
                    "family_assignment": {
                        "interpro_family_id": "IPR000001",
                        "basis": "reviewed family assignment",
                    },
                },
            )
            register_atlas_homologs(run_dir, update)
            atlas_metadata = load_candidate_registry(
                run_dir
            )["candidates"]["MGYP000932894257"]["metadata"]
            self.assertEqual(atlas_metadata["screening_family_id"], "IPR000001")

            screened = screen_candidates(run_dir, ["MGYP000932894257"])
            self.assertEqual(screened["status"], "success")
            self.assertEqual(
                current_candidate_status(
                    load_candidate_registry(run_dir)["candidates"]["MGYP000932894257"]
                ),
                "screened",
            )

            result = evaluate_candidate(run_dir, "MGYP000932894257")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["data"]["evaluation_scope"], "exploratory_structural_lead")
            self.assertIsNone(result["data"]["is_enzyme"])
            self.assertIn("UNANNOTATED METAGENOMIC LEAD", result["data"]["critical_warnings"][0])

    def test_auto_pipeline_advances_atlas_hit(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
            update_candidate_registry(
                run_dir, "MGYP000932894257", "discovered",
                metadata={
                    "is_metagenomic": True,
                    "sequence": "MSEQUENCE",
                    "screening_family_id": "IPR000001",
                    "interpro_ids": ["IPR000001"],
                    "structural_homology": {"probability": 0.95},
                },
            )
            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._reviewed_member_count",
                return_value=10,
            ), mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._fold_fg_max",
                return_value=None,
            ):
                auto_screen_candidates(run_dir)
            self.assertEqual(_auto_evaluate_all(run_dir), 1)
            candidate = load_candidate_registry(run_dir)["candidates"]["MGYP000932894257"]
            self.assertEqual(current_candidate_status(candidate), "evaluated")


if __name__ == "__main__":
    unittest.main()
