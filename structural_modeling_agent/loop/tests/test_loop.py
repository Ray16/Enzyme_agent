"""Unit tests for the active-search loop: reward, calibration, acquisition."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reward as R  # noqa: E402
import calibration as C  # noqa: E402
import acquisition as ACQ  # noqa: E402


class TestReward(unittest.TestCase):
    def test_product_when_all_present(self):
        r = R.combine_reward(mechanism=0.5, proxy=0.4, developability=0.5)
        self.assertFalse(r.abstained)
        self.assertAlmostEqual(r.reward, 0.1)

    def test_none_factor_abstains(self):
        for kw in ({"mechanism": None}, {"proxy": None}, {"developability": None}):
            args = {"mechanism": 0.9, "proxy": 0.9, "developability": 0.9, **kw}
            r = R.combine_reward(**args)
            self.assertTrue(r.abstained)
            self.assertIsNone(r.reward)

    def test_zero_factor_scores_zero_not_abstained(self):
        r = R.combine_reward(mechanism=0.9, proxy=0.9, developability=0.0)
        self.assertFalse(r.abstained)
        self.assertEqual(r.reward, 0.0)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            R.combine_reward(mechanism=1.2, proxy=0.5, developability=0.5)


class TestCalibration(unittest.TestCase):
    def _ledger(self, pairs, family="F1", abstained_pair=None):
        led = C.CalibrationLedger()
        for i, (p, a) in enumerate(pairs):
            led.add(C.CalibrationRecord(candidate_id=f"c{i}", proxy_score=p,
                                        assay_value=a, family=family))
        if abstained_pair:
            led.add(C.CalibrationRecord(candidate_id="abs", proxy_score=abstained_pair[0],
                                        assay_value=abstained_pair[1], family=family,
                                        abstained=True))
        return led

    def test_perfect_correlation(self):
        led = self._ledger([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])
        corr = led.correlation("F1")
        self.assertEqual(corr["n"], 3)
        self.assertAlmostEqual(corr["spearman"], 1.0)

    def test_abstained_excluded_from_correlation(self):
        led = self._ledger([(0.1, 0.1), (0.9, 0.9)], abstained_pair=(0.9, 0.0))
        self.assertEqual(led.correlation("F1")["n"], 2)

    def test_uncertainty_drops_with_good_data(self):
        weak = C.CalibrationLedger().surrogate_uncertainty("F1")  # empty
        strong = self._ledger([(0.1, 0.1), (0.3, 0.3), (0.5, 0.5),
                               (0.7, 0.7), (0.9, 0.9)]).surrogate_uncertainty("F1")
        self.assertEqual(weak, 1.0)
        self.assertLess(strong, weak)

    def test_calibrated_proxy_shrinks_to_prior_without_data(self):
        cal = C.CalibrationLedger().calibrated_proxy(0.9, "F1")
        self.assertEqual(cal["uncertainty"], 1.0)
        self.assertAlmostEqual(cal["estimate"], C.PRIOR_MEAN)

    def test_calibrated_proxy_trusts_with_data(self):
        led = self._ledger([(0.1, 0.1), (0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (0.9, 0.9)])
        cal = led.calibrated_proxy(0.9, "F1")
        self.assertGreater(cal["estimate"], C.PRIOR_MEAN)  # pulled toward the raw high score

    def test_roundtrip_json(self):
        led = self._ledger([(0.2, 0.3)])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            led.to_json(path)
            back = C.CalibrationLedger.from_json(path)
            self.assertEqual(len(back.records), 1)
            self.assertEqual(back.records[0].candidate_id, "c0")
        finally:
            os.unlink(path)


class TestAcquisition(unittest.TestCase):
    def test_abstained_never_selected(self):
        cands = [
            ACQ.Candidate(id="ok", reward=0.3, uncertainty=0.1, novelty=0.1),
            ACQ.Candidate(id="abs", reward=None, abstained=True),
        ]
        batch = ACQ.select_batch(cands, k=5)
        self.assertEqual([c.id for c in batch], ["ok"])

    def test_exploration_bonus_orders_ties(self):
        cands = [
            ACQ.Candidate(id="safe", reward=0.3, uncertainty=0.0, novelty=0.0),
            ACQ.Candidate(id="novel", reward=0.3, uncertainty=0.8, novelty=0.8),
        ]
        batch = ACQ.select_batch(cands, k=1, beta=0.5)
        self.assertEqual(batch[0].id, "novel")

    def test_beta_zero_is_pure_exploit(self):
        cands = [
            ACQ.Candidate(id="hi", reward=0.6, uncertainty=0.0, novelty=0.0),
            ACQ.Candidate(id="novel", reward=0.3, uncertainty=1.0, novelty=1.0),
        ]
        batch = ACQ.select_batch(cands, k=1, beta=0.0)
        self.assertEqual(batch[0].id, "hi")

    def test_novelty_from_similarity(self):
        self.assertEqual(ACQ.novelty_from_similarity([]), 1.0)
        self.assertAlmostEqual(ACQ.novelty_from_similarity([0.3, 0.9]), 0.1)


if __name__ == "__main__":
    unittest.main()
