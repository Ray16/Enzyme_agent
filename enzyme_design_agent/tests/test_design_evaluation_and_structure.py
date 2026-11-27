"""Tests for mmCIF/PDB structure parsing fixes, evaluation tools, and thread caps."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_design_agent.saturation.results import aggregate
from enzyme_design_agent.structure import load_structure
from enzyme_design_agent.tools import _runtime
from enzyme_design_agent.tools import alphaprotein_novo as apn
from enzyme_design_agent.tools import disco_design as dc
from enzyme_design_agent.tools.design_evaluation import (
    audit_design_contacts,
    summarize_saturation_campaign,
)


def _write_sample_cif(path: Path) -> Path:
    path.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "ATOM   1 N N   . LYS A 1 ? 0.000 0.000 0.000 1\n"
        "ATOM   2 C CA  . LYS A 1 ? 1.450 0.000 0.000 1\n"
        "ATOM   3 N NZ  . LYS A 1 ? 3.000 0.000 0.000 1\n"
        "HETATM 4 O O1  . LIG L 1 ? 6.200 0.000 0.000 1\n"
        "HETATM 5 MG MG . MG  M 1 ? 8.000 0.000 0.000 1\n"
        "ATOM   6 N N   . LYS A 1 ? 99.000 99.000 99.000 2\n"
    )
    return path


class StructureParserFixesTest(unittest.TestCase):
    def test_loads_mmcif_model_1_only_with_atom_names_and_elements(self):
        with tempfile.TemporaryDirectory() as tmp:
            cif = _write_sample_cif(Path(tmp) / "design.cif")
            st = load_structure(cif)
            self.assertEqual("K", st.sequence("A"))
            self.assertEqual(3, st.residues[0].xyz.shape[0])
            self.assertEqual(("N", "CA", "NZ"), st.residues[0].atom_names)
            self.assertEqual(["L", "M"], [c.id for c in st.components])
            mg = st.component("M")
            self.assertEqual(("MG",), mg.elements)

    def test_pdb_stops_at_endmdl_and_parses_two_letter_metals(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "multi.pdb"
            pdb.write_text(
                "MODEL        1\n"
                "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 80.00           C\n"
                "HETATM    2 MG    MG M   1       4.000   2.000   3.000  1.00 80.00            \n"
                "HETATM    3 HG    HG N   1       5.000   2.000   3.000  1.00 80.00            \n"
                "ENDMDL\n"
                "MODEL        2\n"
                "ATOM      1  CA  ALA A   1      99.000  99.000  99.000  1.00 80.00           C\n"
                "ENDMDL\n"
            )
            st = load_structure(pdb)
            self.assertEqual(1, st.residues[0].xyz.shape[0])
            self.assertEqual(("MG",), st.component("M").elements)
            # HG (mercury) without element column must not be dropped as hydrogen.
            self.assertIsNotNone(st.component("N"))
            self.assertEqual(("HG",), st.component("N").elements)


class SaturationResultsAndEvalToolsTest(unittest.TestCase):
    def test_aggregate_excludes_bool_and_nonfinite(self):
        rows = [
            {"id": "v1", "mode": "holo", "iptm": 0.8},
            {"id": "v1", "mode": "holo", "iptm": True},
            {"id": "v1", "mode": "holo", "iptm": math.nan},
            {"id": "v1", "mode": "holo", "iptm": math.inf},
            {"id": "v1", "mode": "holo", "iptm": 0.6},
        ]
        agg = aggregate(rows, value_key="iptm", expect=2)
        self.assertEqual(1, len(agg))
        self.assertEqual(2, agg[0]["n_models"])
        self.assertEqual(0.7, agg[0]["iptm_mean"])
        self.assertEqual(1, agg[0]["complete"])

    def test_summarize_saturation_campaign_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "results" / "cand_WT" / "boltz_results_cand_WT" / "predictions" / "cand_WT"
            pred.mkdir(parents=True)
            for idx, iptm in enumerate((0.82, 0.86)):
                (pred / f"confidence_cand_WT_model_{idx}.json").write_text(
                    json.dumps({"iptm": iptm, "complex_plddt": 0.9})
                )
                (pred / f"cand_WT_model_{idx}.pdb").write_text("ATOM\n")
            res = summarize_saturation_campaign(str(root), expect_models=2)
            self.assertEqual("success", res["status"])
            self.assertEqual(2, res["data"]["n_models"])
            self.assertEqual(1, res["data"]["n_complete_groups"])
            self.assertEqual(0.84, res["data"]["summary_rows"][0]["iptm_mean"])

    def test_audit_design_contacts_on_mmcif(self):
        with tempfile.TemporaryDirectory() as tmp:
            cif = _write_sample_cif(Path(tmp) / "design.cif")
            res = audit_design_contacts(
                structure_path=str(cif),
                target_groups=[{
                    "group_id": "carb",
                    "chain": "L",
                    "resi": "1",
                    "acceptors": ["O1"],
                    "formal_charge": -1,
                    "provenance": "synthetic carboxylate oxygen",
                }],
                protein_chain="A",
            )
            self.assertEqual("success", res["status"])
            self.assertEqual(1, res["data"]["n_accepted"])
            self.assertEqual(1, res["data"]["distinct_accepted_residues"])
            self.assertEqual("potential_ionic", res["data"]["contacts"][0]["kind"])


class ResourceSafetyCapsTest(unittest.TestCase):
    def test_disco_and_apnovo_cap_all_five_thread_vars(self):
        required = (
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        disco_env = dc._subprocess_env(["0"], None, False)
        apn_env = apn._subprocess_env(["0"])
        for key in required:
            self.assertIn(key, disco_env)
            self.assertIn(key, apn_env)
        self.assertEqual("false", disco_env["TOKENIZERS_PARALLELISM"])
        self.assertEqual("false", apn_env["TOKENIZERS_PARALLELISM"])

    def test_disco_does_not_double_release_gpu_on_launch_exception(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(dc, "_readiness", return_value={"ready": True, "blocking": [], "optional": [], "paths": {}}), \
             mock.patch.object(dc, "_local_gpu_capability", return_value=8.0), \
             mock.patch.object(dc, "_acquire_gpus", return_value=["2"]), \
             mock.patch.object(dc, "_release_gpus") as release, \
             mock.patch.object(dc.subprocess, "run", side_effect=OSError("no such binary")):
            res = dc.design_enzyme_with_disco(lengths=[100], num_seeds=1, run_dir=tmp)
        self.assertEqual("error", res["status"])
        self.assertEqual(1, release.call_count)

    def test_runtime_helpers_strip_cuda_and_check_liveness(self):
        env = _runtime.base_subprocess_env(
            2,
            cuda_visible_devices="1",
            base={"LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:/opt/custom/lib"},
        )
        self.assertEqual("1", env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("/opt/custom/lib", env["LD_LIBRARY_PATH"])
        for key in _runtime.THREAD_ENV_KEYS:
            self.assertEqual("2", env[key])
        import os
        ticks = _runtime.process_start_ticks(os.getpid())
        self.assertTrue(_runtime.is_process_alive(os.getpid(), ticks))
        self.assertFalse(_runtime.is_process_alive(os.getpid(), "wrong-ticks"))


if __name__ == "__main__":
    unittest.main()
