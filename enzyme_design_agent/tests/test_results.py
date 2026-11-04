"""Campaign bookkeeping: discovery, dedup, completeness, and metric delegation."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from enzyme_design_agent.saturation import results as R


def _write_model(d: Path, vid: str, sample: int, pc=None, **scalars):
    d.mkdir(parents=True, exist_ok=True)
    payload = dict(scalars)
    if pc is not None:
        payload["pair_chains_iptm"] = pc
    (d / f"confidence_{vid}_model_{sample}.json").write_text(json.dumps(payload))
    return d / f"confidence_{vid}_model_{sample}.json"


class DiscoverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovers_id_and_sample_from_the_filename(self):
        _write_model(self.root / "whatever" / "nested", "e_S2C", 7)
        c = R.discover(self.root)
        self.assertEqual(1, len(c.models))
        self.assertEqual("e_S2C", c.models[0].variant_id)
        self.assertEqual(7, c.models[0].sample)

    def test_id_with_underscores_survives(self):
        _write_model(self.root, "enz_arm_S190T", 0)
        self.assertEqual("enz_arm_S190T", R.discover(self.root).models[0].variant_id)

    def test_mode_comes_from_a_path_segment(self):
        _write_model(self.root / "holo" / "x", "v", 0)
        _write_model(self.root / "apo" / "x", "v", 0)
        c = R.discover(self.root, modes=("apo", "holo"))
        self.assertEqual({"apo", "holo"}, {m.mode for m in c.models})

    def test_no_modes_means_one_condition(self):
        _write_model(self.root / "x", "v", 0)
        self.assertEqual("", R.discover(self.root).models[0].mode)

    def test_two_layouts_of_one_model_are_not_double_counted(self):
        _write_model(self.root / "boltz_results_v" / "predictions" / "v", "v", 3)
        _write_model(self.root / "by_seed" / "v" / "s3", "v", 3)
        c = R.discover(self.root)
        self.assertEqual(1, len(c.models))
        self.assertEqual(1, c.duplicates)

    def test_symlinked_subtrees_are_followed(self):
        real = self.root / "real"
        _write_model(real, "v", 0)
        link = self.root / "staged"
        os.symlink(real, link)
        # One model, reachable two ways: found, and counted once.
        c = R.discover(self.root)
        self.assertEqual(1, len(c.models))

    def test_unknown_ids_are_reported_not_merged(self):
        _write_model(self.root, "known", 0)
        _write_model(self.root, "stranger", 0)
        c = R.discover(self.root, known_ids={"known"})
        self.assertEqual(["stranger"], list(c.unknown_ids))
        self.assertEqual(["known"], [m.variant_id for m in c.models])

    def test_structure_beside_the_json_is_linked(self):
        p = _write_model(self.root, "v", 0)
        (p.parent / "v_model_0.pdb").write_text("END\n")
        self.assertTrue(R.discover(self.root).models[0].structure_path.endswith(".pdb"))

    def test_groups_are_keyed_by_variant_and_mode(self):
        for s in range(3):
            _write_model(self.root / "apo", "v", s)
        c = R.discover(self.root, modes=("apo",))
        self.assertEqual({("v", "apo")}, set(c.groups()))
        self.assertEqual([0, 1, 2], [m.sample for m in c.groups()[("v", "apo")]])


class ConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_chain_pair_takes_the_minimum_direction(self):
        _write_model(self.root, "v", 0,
                     pc={"0": {"1": 0.99}, "1": {"0": 0.81}}, iptm=0.95)
        ref = R.discover(self.root).models[0]
        row = R.read_confidence(ref, chain_pairs={"ps_iptm": ("0", "1")})
        self.assertEqual(0.81, row["ps_iptm"])
        self.assertAlmostEqual(0.18, row["ps_iptm_asym"], places=6)

    def test_missing_pair_is_none_not_an_error(self):
        _write_model(self.root, "v", 0, pc={})
        ref = R.discover(self.root).models[0]
        self.assertIsNone(R.read_confidence(ref, {"ps_iptm": ("0", "1")})["ps_iptm"])

    def test_no_default_chain_pair_is_assumed(self):
        """Which chain is the substrate is campaign knowledge, not a default."""
        _write_model(self.root, "v", 0, pc={"0": {"1": 0.9}})
        ref = R.discover(self.root).models[0]
        self.assertNotIn("ps_iptm", R.read_confidence(ref))

    def test_unreadable_json_reports_an_error_row(self):
        (self.root / "confidence_v_model_0.json").write_text("{not json")
        ref = R.discover(self.root).models[0]
        self.assertIn("error", R.read_confidence(ref))


class AggregateTest(unittest.TestCase):
    def test_aggregates_whichever_column_is_asked_for(self):
        rows = [{"id": "v", "mode": "a", "x": 1.0}, {"id": "v", "mode": "a", "x": 3.0}]
        out = R.aggregate(rows, "x")
        self.assertEqual(2, out[0]["n_models"])
        self.assertEqual(2.0, out[0]["x_mean"])
        self.assertEqual(3.0, out[0]["x_max"])

    def test_sd_is_blank_for_a_single_model(self):
        out = R.aggregate([{"id": "v", "mode": "a", "x": 1.0}], "x")
        self.assertEqual("", out[0]["x_sd"])

    def test_completeness_is_carried(self):
        rows = [{"id": "v", "mode": "a", "x": 1.0}]
        self.assertEqual(0, R.aggregate(rows, "x", expect=50)[0]["complete"])
        self.assertEqual(1, R.aggregate(rows, "x", expect=1)[0]["complete"])

    def test_non_numeric_values_do_not_inflate_n_models(self):
        """A missing metric must not be counted as a measured one."""
        rows = [{"id": "v", "mode": "a", "x": 1.0}, {"id": "v", "mode": "a", "x": None}]
        self.assertEqual(1, R.aggregate(rows, "x")[0]["n_models"])

    def test_groups_split_by_mode(self):
        rows = [{"id": "v", "mode": "a", "x": 1.0}, {"id": "v", "mode": "b", "x": 2.0}]
        self.assertEqual(2, len(R.aggregate(rows, "x")))


class ScoreDelegationTest(unittest.TestCase):
    """The metric comes from the structural modeling pipeline, not from here."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_structures_are_reported_not_dropped(self):
        _write_model(self.root, "v", 0)                 # JSON only, no .pdb
        c = R.discover(self.root)
        out = R.score_groups(c, resolver=lambda pose: None, min_poses=1)
        self.assertIn("error", out[("v", "")])
        self.assertEqual(0, out[("v", "")]["n_poses"])

    def test_unresolvable_poses_are_reported_not_dropped(self):
        from enzyme_design_agent.tests._synthetic import write_structure

        p = _write_model(self.root, "v", 0)
        write_structure(p.parent / "v_model_0.pdb")
        c = R.discover(self.root)
        out = R.score_groups(c, resolver=lambda pose: None)
        self.assertIn("resolver returned no reactive site", out[("v", "")]["error"])

    def test_scores_come_from_the_proxy_library(self):
        from structural_modeling_agent.proxies import ReactiveSite

        from enzyme_design_agent.tests._synthetic import write_structure

        for s in range(3):
            p = _write_model(self.root, "v", s)
            write_structure(p.parent / f"v_model_{s}.pdb")
        c = R.discover(self.root)

        def resolver(pose):
            lig = pose.ligands["L"]
            return ReactiveSite(nucleophile=pose.ca[0], electrophile=lig[0][2])

        out = R.score_groups(c, resolver=resolver)[("v", "")]
        self.assertEqual(3, out["n_poses"])
        # The proxy ids are the library's, not names invented in this package.
        self.assertIn("approach_distance", out)


if __name__ == "__main__":
    unittest.main()
