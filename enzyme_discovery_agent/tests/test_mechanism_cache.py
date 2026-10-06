"""Tests for the family-keyed reaction-coordinate cache (retrieve + update).

Generic: uses placeholder family ids and abstract steps; no specific reaction.
"""
import sys
import unittest
from pathlib import Path

# mechanism_cache is stdlib-only, so import it directly by path (no package install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from mechanism_cache import MechanismCache  # noqa: E402

STEPS = [
    {"index": 1, "name": "step-a", "species_present": ["X", "Y"], "species_consumed": ["Y"]},
    {"index": 2, "name": "step-b", "species_present": ["Z"], "species_consumed": ["Z"]},
]


class MechanismCacheTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "cache.json"

    def _cache(self):
        return MechanismCache(self.path)

    def test_miss_then_retrieve_after_update(self):
        c = self._cache()
        self.assertIsNone(c.get(["IPR0001"]))            # retrieve: miss
        self.assertTrue(c.put(["IPR0001"], mechanism_class="mech-1", steps=STEPS, grounding_score=5.0))
        rec = MechanismCache(self.path).get(["IPR0001"])  # persisted + retrievable
        self.assertIsNotNone(rec)
        self.assertEqual(rec["mechanism_class"], "mech-1")
        self.assertEqual(len(rec["steps"]), 2)
        self.assertEqual(rec["version"], 1)

    def test_transfer_to_homolog_sharing_a_family(self):
        c = self._cache()
        c.put(["IPR0001", "IPR0002"], mechanism_class="mech-1", steps=STEPS, grounding_score=5.0)
        # An unannotated homolog that shares one family id retrieves the coordinate.
        self.assertIsNotNone(c.get(["IPR9999", "IPR0002"]))

    def test_update_does_not_regress_better_grounding(self):
        c = self._cache()
        c.put(["IPR0001"], mechanism_class="good", steps=STEPS, grounding_score=8.0)
        wrote = c.put(["IPR0001"], mechanism_class="worse", steps=STEPS[:1], grounding_score=2.0)
        self.assertFalse(wrote)
        self.assertEqual(c.get(["IPR0001"])["mechanism_class"], "good")

    def test_update_archives_prior_version(self):
        c = self._cache()
        c.put(["IPR0001"], mechanism_class="v1", steps=STEPS, grounding_score=3.0)
        c.put(["IPR0001"], mechanism_class="v2", steps=STEPS, grounding_score=5.0)
        rec = c.get(["IPR0001"])
        self.assertEqual(rec["mechanism_class"], "v2")
        self.assertEqual(rec["version"], 2)
        self.assertTrue(any(h.get("mechanism_class") == "v1" for h in rec.get("history", [])))

    def test_force_overwrites(self):
        c = self._cache()
        c.put(["IPR0001"], mechanism_class="good", steps=STEPS, grounding_score=8.0)
        self.assertTrue(c.put(["IPR0001"], mechanism_class="forced", steps=STEPS, grounding_score=1.0, force=True))
        self.assertEqual(c.get(["IPR0001"])["mechanism_class"], "forced")

    def test_export_groups_families_by_mechanism(self):
        c = self._cache()
        c.put(["IPR0001"], mechanism_class="mech-1", steps=STEPS, grounding_score=5.0)
        c.put(["IPR0002"], mechanism_class="mech-1", steps=STEPS, grounding_score=5.0)
        exported = c.export_mechanisms()["mechanisms"]
        self.assertIn("mech-1", exported)
        self.assertCountEqual(exported["mech-1"]["keys"]["families"], ["IPR0001", "IPR0002"])

    def test_stale_instances_merge_instead_of_losing_updates(self):
        first = self._cache()
        second = self._cache()
        first.put(["IPR0001"], mechanism_class="mech-1", steps=STEPS, grounding_score=5.0)
        second.put(["IPR0002"], mechanism_class="mech-2", steps=STEPS, grounding_score=5.0)
        families = self._cache().doc["families"]
        self.assertCountEqual(families, ["IPR0001", "IPR0002"])

    def test_stale_save_does_not_erase_newer_updates(self):
        stale = self._cache()
        current = self._cache()
        current.put(["IPR0001"], mechanism_class="mech-1", steps=STEPS, grounding_score=5.0)
        stale.save()
        self.assertIn("IPR0001", self._cache().doc["families"])

    def test_stale_export_reloads_newer_updates(self):
        stale = self._cache()
        current = self._cache()
        current.put(["IPR0001"], mechanism_class="mech-1", steps=STEPS, grounding_score=5.0)
        exported = stale.export_mechanisms()["mechanisms"]
        self.assertIn("IPR0001", exported["mech-1"]["keys"]["families"])


if __name__ == "__main__":
    unittest.main()
