"""The two saturation tools as the agent calls them: through the registry."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from enzyme_design_agent.bootstrap import register_design_tools
from enzyme_design_agent.tests._synthetic import SEQUENCE, write_structure
from enzyme_discovery_agent.tool_registry import execute_tool


def _run(name, args, run_dir):
    return execute_tool(name, args, injected_args={"run_dir": str(run_dir)})


class ScanPocketToolTest(unittest.TestCase):
    def setUp(self):
        register_design_tools()
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.pdb = write_structure(self.d / "holo.pdb")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_positions_and_writes_a_tsv(self):
        r = _run("scan_design_pocket", {"structure_path": str(self.pdb)}, self.d)
        self.assertEqual("success", r["status"])
        self.assertEqual(2, r["data"]["n_positions"])
        self.assertTrue(Path(r["data"]["positions_tsv"]).is_file())

    def test_reports_the_excluded_stray_component(self):
        r = _run("scan_design_pocket", {"structure_path": str(self.pdb)}, self.d)
        ex = r["data"]["excluded_components"]
        self.assertEqual(["N"], [e["component"] for e in ex])
        self.assertIn("excluded", r["summary"])

    def test_scoping_to_one_ligand_loses_the_cofactor_position(self):
        whole = _run("scan_design_pocket", {"structure_path": str(self.pdb)}, self.d)
        one = _run("scan_design_pocket",
                   {"structure_path": str(self.pdb), "components": "L"}, self.d)
        self.assertGreater(whole["data"]["n_positions"], one["data"]["n_positions"])

    def test_missing_file_is_an_error_not_a_crash(self):
        r = _run("scan_design_pocket", {"structure_path": str(self.d / "nope.pdb")}, self.d)
        self.assertEqual("error", r["status"])

    def test_bad_component_lists_what_is_available(self):
        r = _run("scan_design_pocket",
                 {"structure_path": str(self.pdb), "components": "ZZ"}, self.d)
        self.assertEqual("error", r["status"])
        self.assertIn("L", r["data"]["available_components"])


class BuildLibraryToolTest(unittest.TestCase):
    def setUp(self):
        register_design_tools()
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.pdb = write_structure(self.d / "holo.pdb")

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_from_a_comma_list(self):
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb), "positions": "2,3", "label": "e"},
                 self.d)
        self.assertEqual("success", r["status"])
        self.assertEqual(1 + 2 * 19, r["data"]["n_variants"])
        self.assertTrue(Path(r["data"]["variants_fasta"]).is_file())

    def test_builds_from_a_positions_tsv(self):
        scan = _run("scan_design_pocket", {"structure_path": str(self.pdb)}, self.d)
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb),
                  "positions": scan["data"]["positions_tsv"]}, self.d)
        self.assertEqual("success", r["status"])
        self.assertEqual(2, r["data"]["n_positions"])

    def test_modes_multiply_the_job_count(self):
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb), "positions": "2",
                  "modes": "apo,holo", "n_samples": 50}, self.d)
        self.assertEqual(2 * 20, r["data"]["n_jobs"])
        self.assertEqual(50 * 2 * 20, r["data"]["n_models"])

    def test_annotation_is_recorded_and_the_position_stays(self):
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb), "positions": "2,3",
                  "annotate_positions": json.dumps({"2": "catalytic"})}, self.d)
        self.assertEqual([2], r["data"]["annotated_positions"])
        self.assertEqual(1 + 2 * 19, r["data"]["n_variants"])
        rows = list(csv.DictReader(Path(r["data"]["variants_tsv"]).open(), delimiter="\t"))
        self.assertTrue(any(row["annotation"] == "catalytic" for row in rows))

    def test_sequences_in_the_tsv_match_the_structure(self):
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb), "positions": "2"}, self.d)
        rows = list(csv.DictReader(Path(r["data"]["variants_tsv"]).open(), delimiter="\t"))
        wt = next(row for row in rows if row["kind"] == "wt")
        self.assertEqual(SEQUENCE, wt["seq"])

    def test_position_outside_the_sequence_is_refused(self):
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb), "positions": "999"}, self.d)
        self.assertEqual("error", r["status"])
        self.assertIn("outside the sequence", r["summary"])

    def test_bad_annotation_json_is_refused(self):
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb), "positions": "2",
                  "annotate_positions": "{oops"}, self.d)
        self.assertEqual("error", r["status"])

    def test_empty_positions_is_refused(self):
        r = _run("build_saturation_library",
                 {"structure_path": str(self.pdb), "positions": ""}, self.d)
        self.assertEqual("error", r["status"])


if __name__ == "__main__":
    unittest.main()
