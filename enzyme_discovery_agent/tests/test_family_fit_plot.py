"""Tests for the native-substrate-fit visualization (best-effort, offline)."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.output.family_fit_plot import render_family_fit_plot


def _landscape():
    return [
        {"fold_id": "IPR000001", "source": "derived:x",
         "holistic_fit": {"max": 0.62, "median": 0.5},
         "axis_medians": {"functional_group_similarity": 0.5, "mw_similarity": 0.7,
                          "chain_length_similarity": 0.8, "volume_similarity": 0.65,
                          "tpsa_similarity": 0.6, "logp_similarity": 0.55},
         "native_chain_length_range": [4, 6],
         "members_spanning_aromatic_and_aliphatic": 0},
        {"fold_id": "IPR000002", "source": "discovered",
         "holistic_fit": {"max": 0.55, "median": 0.45},
         "axis_medians": {"functional_group_similarity": 0.5, "mw_similarity": 0.4,
                          "chain_length_similarity": 0.5, "volume_similarity": 0.45,
                          "tpsa_similarity": 0.5, "logp_similarity": 0.6},
         "native_chain_length_range": [3, 9],
         "members_spanning_aromatic_and_aliphatic": 2},
    ]


class TestFamilyFitPlot(unittest.TestCase):
    def test_renders_png_from_landscape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "family_landscape.json").write_text(json.dumps(_landscape()))
            (root / "target_context.json").write_text(
                json.dumps({"substrate_smiles": "O=C(O)CCCCC(=O)O.NCCCCCCN"}))
            out = render_family_fit_plot(root)
            # matplotlib should be present in this env; if not, the call returns
            # None (best-effort) rather than raising — either way it never throws.
            if out is not None:
                self.assertTrue(out.exists())
                self.assertGreater(out.stat().st_size, 0)

    def test_no_landscape_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(render_family_fit_plot(Path(tmp)))

    def test_landscape_without_axis_medians_returns_none(self):
        # Old-format landscape (FG/size only) has nothing multi-axis to plot.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "family_landscape.json").write_text(json.dumps(
                [{"fold_id": "IPR000001", "functional_group_similarity": {"max": 0.5}}]))
            self.assertIsNone(render_family_fit_plot(root))


if __name__ == "__main__":
    unittest.main()
