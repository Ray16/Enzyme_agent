import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from proxy_policy import (
    mechanism_from_reaction,
    proxy_mechanism_for_family,
    resolve_mechanism,
    structural_support_tier,
)


def _load_angle_module():
    path = ROOT / "scripts" / "angle_gate_proxy.py"
    spec = importlib.util.spec_from_file_location("angle_gate_proxy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prioritization_module():
    path = ROOT / "scripts" / "build_candidate_prioritization.py"
    spec = importlib.util.spec_from_file_location("prioritization_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProxyPolicyTests(unittest.TestCase):
    def test_family_resolution_is_fail_closed_and_evidence_labeled(self):
        anl = proxy_mechanism_for_family("AMP-binding enzyme")
        self.assertEqual((anl.phosphate, anl.evidence_scope), ("PA", "panel_validated"))

        nis = proxy_mechanism_for_family("Aerobactin siderophore biosynthesis, IucA/IucC-like")
        self.assertEqual((nis.phosphate, nis.evidence_scope), ("PA", "mechanistically_transferred"))

        grasp = proxy_mechanism_for_family("ATP-grasp fold")
        self.assertEqual((grasp.phosphate, grasp.evidence_scope), ("PG", "mechanistically_transferred"))
        self.assertIsNone(proxy_mechanism_for_family("uncharacterized enzyme"))

    def test_support_tier_distinguishes_sparse_from_negative_evidence(self):
        self.assertEqual(structural_support_tier(170.0, 4, "assigned"), "inconclusive")
        self.assertEqual(structural_support_tier(149.0, 40, "assigned"), "weak geometric support")
        self.assertEqual(structural_support_tier(160.0, 40, "assigned"), "compatible")
        self.assertEqual(structural_support_tier(170.0, 40, "assigned"), "strong geometric support")

    def test_score_reports_sampling_and_uncertainty_without_iptm_filter(self):
        module = _load_angle_module()
        geometries = [
            {"d_pa": d, "ang_pa": angle}
            for d, angle in [
                (3.2, 160.0),
                (3.3, 162.0),
                (3.4, 164.0),
                (3.5, 166.0),
                (3.6, 168.0),
                (4.5, 179.0),
            ]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp)
            pdbs = []
            for index in range(len(geometries)):
                path = pred / f"x_model_{index}.pdb"
                path.touch()
                pdbs.append(str(path))
            with patch.object(module.s, "geom_model", side_effect=geometries):
                result = module.score_config(str(pred), "pa")

        self.assertEqual(result["n_models"], 6)
        self.assertEqual(result["n_gated"], 5)
        self.assertEqual(result["gated_fraction"], 0.833)
        self.assertTrue(result["docking_ok"])
        self.assertEqual(result["mean_inline_angle"], 164.0)
        self.assertLessEqual(result["angle_ci90_low"], 164.0)
        self.assertGreaterEqual(result["angle_ci90_high"], 164.0)
        self.assertGreater(result["prob_mean_ge_150"], 0.99)

    def test_prioritization_preserves_uncertainty_and_evidence_scope(self):
        module = _load_prioritization_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reps = root / "representatives.csv"
            angles = root / "angles.csv"
            output = root / "prioritization.csv"
            reps.write_text(
                "rank,tier_label,accession,gene,protein_name,interpro_family\n"
                "1,strong lead,A1,g,p,AMP-binding enzyme\n"
                "2,worth testing,A2,g,p,uncharacterized enzyme\n"
            )
            angles.write_text(
                "id,proxy_version,phosphate,n_models,n_gated,gated_fraction,"
                "mean_inline_angle,angle_std,angle_ci90_low,angle_ci90_high,"
                "prob_mean_ge_150,prob_mean_ge_165,inline_frac_gated,mean_iptm\n"
                "A1,step1-angle-v2,PA,50,20,0.4,166.0,5.0,164.1,167.9,"
                "1.0,0.82,0.95,0.7\n"
                "A2,step1-angle-v2,,50,0,0.0,,,,,,,0.0,0.7\n"
            )
            with patch.object(
                sys,
                "argv",
                [
                    "build_candidate_prioritization.py",
                    "--representatives",
                    str(reps),
                    "--angles",
                    str(angles),
                    "--out",
                    str(output),
                ],
            ):
                module.main()
            with output.open() as handle:
                rows = {row["accession"]: row for row in csv.DictReader(handle)}

        self.assertEqual(rows["A1"]["proxy_evidence_scope"], "panel_validated")
        self.assertEqual(rows["A1"]["angle_ci90_low"], "164.1")
        self.assertEqual(rows["A1"]["structural_tier"], "strong geometric support")
        self.assertEqual(rows["A2"]["mechanism_status"], "unresolved")
        self.assertEqual(rows["A2"]["structural_tier"], "inconclusive")
        self.assertEqual(rows["A2"]["struct_rank_within_family"], "")


if __name__ == "__main__":
    unittest.main()
