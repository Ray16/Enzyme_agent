"""Fail-closed tests for structural cofold input generation."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_step1_inputs.py"
SPEC = importlib.util.spec_from_file_location("build_step1_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestReadiness(unittest.TestCase):
    def test_unreviewed_charge_change_is_blocking(self):
        errors = MODULE.readiness_errors({
            "status": "needs_review",
            "unresolved": ["target component 1 modeling state requires review"],
            "geometry": {
                "electrophile": {"atom": "atom 2"},
            },
            "enzyme_state": {"chains": [{"source": "full_sequence"}]},
            "cofactors": [],
            "target_substrates": [{
                "component_index": 1,
                "role": "substrate",
                "include_in_model": True,
                "review_status": "required",
                "modeling_smiles": "[O-]C(=O)CCCCC(=O)[O-]",
                "modeling_representation": {
                    "stereochemistry": {"status": "not_applicable"},
                },
            }],
        })
        self.assertTrue(any("status=" in error for error in errors))
        self.assertTrue(any("unreviewed" in error for error in errors))

    def test_ready_complete_input_passes(self):
        errors = MODULE.readiness_errors({
            "status": "ready",
            "unresolved": [],
            "geometry": {
                "electrophile": {"atom": "atom 2"},
            },
            "enzyme_state": {"chains": [{"source": "full_sequence"}]},
            "cofactors": [],
            "target_substrates": [{
                "component_index": 1,
                "role": "substrate",
                "include_in_model": True,
                "review_status": "reviewed",
                "modeling_smiles": "[O-]C(=O)CCCCC(=O)[O-]",
                "modeling_representation": {
                    "stereochemistry": {"status": "not_applicable"},
                },
            }],
        })
        self.assertEqual(errors, [])

    def test_geometry_is_not_required_to_build_an_unrestrained_cofold(self):
        errors = MODULE.readiness_errors({
            "status": "ready",
            "unresolved": [],
            "geometry": None,
            "enzyme_state": {"chains": [{"source": "full_sequence"}]},
            "cofactors": [],
            "target_substrates": [{
                "component_index": 1,
                "role": "substrate",
                "include_in_model": True,
                "review_status": "not_required",
                "modeling_smiles": "CC(=O)N",
                "modeling_representation": {
                    "stereochemistry": {"status": "not_applicable"},
                },
            }],
        })
        self.assertEqual(errors, [])

    def test_processed_chain_requires_fragment_msa(self):
        with self.assertRaisesRegex(ValueError, "fragment MSA"):
            MODULE.enzyme_chains(
                "AAATTT",
                "/tmp/full.csv",
                [
                    {"source": "sequence_range", "start": 1, "end": 3},
                    {"source": "sequence_range", "start": 4, "end": 6},
                ],
            )

    def test_processed_chain_uses_reviewed_ranges(self):
        chains = MODULE.enzyme_chains(
            "AAATTT",
            "/tmp/full.csv",
            [
                {
                    "source": "sequence_range",
                    "start": 1,
                    "end": 3,
                    "msa": "/tmp/a.csv",
                },
                {
                    "source": "sequence_range",
                    "start": 4,
                    "end": 6,
                    "msa": "/tmp/b.csv",
                },
            ],
        )
        self.assertEqual([chain["sequence"] for chain in chains], ["AAA", "TTT"])


class TestYamlRendering(unittest.TestCase):
    @staticmethod
    def _entity_lines(rendered):
        return [line.strip() for line in rendered.splitlines()]

    def test_intact_enzyme_with_cofactors(self):
        chains = MODULE.enzyme_chains(
            "ACDE", "/tmp/full.csv", [{"source": "full_sequence"}],
        )
        rendered = MODULE.yaml_for(
            chains,
            [{"ccd": "ATP", "count": 1}, {"ccd": "MG", "count": 2}],
            [{
                "modeling_smiles": "[O-]C(=O)CC(=O)[O-]",
                "affinity_target": True,
            }],
            include_affinity=True,
        )
        lines = self._entity_lines(rendered)
        self.assertIn('sequence: "ACDE"', lines)
        self.assertEqual(lines.count('ccd: "ATP"'), 1)
        self.assertEqual(lines.count('ccd: "MG"'), 2)
        self.assertIn('smiles: "[O-]C(=O)CC(=O)[O-]"', lines)
        self.assertIn("binder: E", lines)

    def test_processed_enzyme_without_cofactors(self):
        chains = MODULE.enzyme_chains(
            "AAATTT",
            "/tmp/full.csv",
            [
                {
                    "source": "sequence_range",
                    "start": 1,
                    "end": 3,
                    "msa": "/tmp/a.csv",
                },
                {
                    "source": "sequence_range",
                    "start": 4,
                    "end": 6,
                    "msa": "/tmp/b.csv",
                },
            ],
        )
        rendered = MODULE.yaml_for(
            chains,
            [],
            [{"modeling_smiles": "CC(=O)NCC", "affinity_target": True}],
        )
        lines = self._entity_lines(rendered)
        self.assertEqual(lines.count("- protein:"), 2)
        self.assertIn('sequence: "AAA"', lines)
        self.assertIn('sequence: "TTT"', lines)
        self.assertIn('msa: "/tmp/a.csv"', lines)
        self.assertIn('msa: "/tmp/b.csv"', lines)
        self.assertIn('smiles: "CC(=O)NCC"', lines)
        self.assertFalse(any(line.startswith("ccd:") for line in lines))

    def test_invalid_cofactor_record_is_blocking(self):
        errors = MODULE.readiness_errors({
            "status": "ready",
            "unresolved": [],
            "enzyme_state": {"chains": [{"source": "full_sequence"}]},
            "cofactors": [{"name": "unknown metal", "count": 1}],
            "target_substrates": [{
                "component_index": 1,
                "role": "substrate",
                "include_in_model": True,
                "review_status": "not_required",
                "modeling_smiles": "CCO",
                "modeling_representation": {
                    "valid": True,
                    "stereochemistry": {"status": "not_applicable"},
                },
            }],
        })
        self.assertIn("a cofactor lacks a CCD identifier", errors)

    def test_malformed_input_cannot_be_overridden(self):
        modeling_input = {
            "status": "needs_review",
            "unresolved": ["cofactor evidence is incomplete"],
            "enzyme_state": {"chains": []},
            "cofactors": [{"name": "unknown metal", "count": 1}],
            "target_substrates": [],
        }
        self.assertTrue(MODULE.execution_shape_errors(modeling_input))


class TestCommandIntegration(unittest.TestCase):
    def test_ready_handoff_builds_yaml_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "candidates.fasta"
            msa_file = root / "candidate.csv"
            msa_index = root / "index.tsv"
            handoff = root / "handoff.json"
            out = root / "out"
            fasta.write_text(">E00001\nACDE\n")
            msa_file.write_text("key,sequence\n0,ACDE\n")
            msa_index.write_text(f"E00001\tunused\t{msa_file}\n")
            handoff.write_text(json.dumps({
                "schema_version": "2.0",
                "candidates": [{
                    "accession": "E00001",
                    "evidence": {
                        "identity": {"accession": "E00001"},
                    },
                    "modeling_input": {
                        "status": "ready",
                        "unresolved": [],
                        "geometry": None,
                        "enzyme_state": {
                            "state": "intact",
                            "chains": [{"source": "full_sequence"}],
                        },
                        "cofactors": [],
                        "target_substrates": [{
                            "component_index": 1,
                            "role": "substrate",
                            "include_in_model": True,
                            "affinity_target": True,
                            "review_status": "not_required",
                            "modeling_smiles": "CCO",
                            "modeling_representation": {
                                "valid": True,
                                "stereochemistry": {"status": "not_applicable"},
                            },
                        }],
                    },
                    "selection": {
                        "source": "evaluated_discovery_candidate",
                        "reason": "selected for structural modeling",
                    },
                }],
            }))

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--handoff",
                    str(handoff),
                    "--fasta",
                    str(fasta),
                    "--msa-index",
                    str(msa_index),
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rendered = (out / "step1_E00001.yaml").read_text()
            manifest = json.loads((out / "campaign_manifest.json").read_text())

        self.assertIn("wrote 1 step-1 YAMLs", completed.stdout)
        self.assertIn('sequence: "ACDE"', rendered)
        self.assertIn('smiles: "CCO"', rendered)
        self.assertFalse(manifest["executions"][0]["exploratory_override"])
        self.assertFalse(manifest["affinity_enabled"])
        self.assertIsNone(manifest["executions"][0]["geometry"])
        self.assertNotIn("properties:", rendered)

    def test_missing_candidate_layer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff.json"
            handoff.write_text(json.dumps({
                "schema_version": "2.0",
                "candidates": [{
                    "accession": "E00001",
                    "modeling_input": {},
                    "selection": {},
                }],
            }))
            with self.assertRaisesRegex(ValueError, "evidence layer"):
                MODULE.load_handoff_modeling(handoff)


if __name__ == "__main__":
    unittest.main()
