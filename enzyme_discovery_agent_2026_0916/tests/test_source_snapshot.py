"""The run's provenance must include a verbatim copy of the code that ran,
under ``scripts/``, plus an aggregate hash for one-shot cross-run comparison.
"""

import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent import provenance


class TestSourceSnapshot(unittest.TestCase):
    def test_snapshot_copies_py_into_scripts(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            info = provenance.snapshot_source_code(run_dir)

            scripts = run_dir / "scripts"
            self.assertTrue(scripts.is_dir())
            # A known module is copied, preserving the package-relative path.
            self.assertTrue((scripts / "provenance.py").exists())
            self.assertTrue((scripts / "output" / "csv_generator.py").exists())
            self.assertGreater(info["source_files"], 20)
            self.assertEqual(info["source_snapshot_dir"], "scripts")
            self.assertRegex(info["source_sha256"], r"^[0-9a-f]{64}$")

    def test_runs_tree_not_copied(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            provenance.snapshot_source_code(run_dir)
            # The (huge, self-referential) runs/ tree must be excluded.
            self.assertFalse((run_dir / "scripts" / "runs").exists())

    def test_hash_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            a = provenance.snapshot_source_code(Path(td1))["source_sha256"]
            b = provenance.snapshot_source_code(Path(td2))["source_sha256"]
            self.assertEqual(a, b, "same source must hash identically across runs")


if __name__ == "__main__":
    unittest.main()
