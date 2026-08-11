"""Tests for the family-coverage guard helper (answer-agnostic).

The guard keeps a run from finalizing while a family the agent itself derived as
promising was never targeted by a search.  These are offline, deterministic tests
of the pure matcher — no network, no answer identities.
"""

import unittest

from enzyme_discovery_agent.tools.mechanism_families import (
    unexplored_promising_families,
)


class TestUnexploredPromisingFamilies(unittest.TestCase):
    def _fam(self, name, descriptors, promise="high"):
        return {"name": name, "descriptors": descriptors, "promise": promise}

    def test_flags_promising_family_never_searched(self):
        families = [
            self._fam("Phosphate-grasp coupling fold", ["carboxylate-phosphate fold"]),
            self._fam("Adenylate-forming coupling fold", ["adenylate-forming"], "medium"),
        ]
        # Only the first family's distinctive token appears in the searches.
        queries = ["carboxylate-phosphate ligase amine coupling"]
        out = unexplored_promising_families(families, queries)
        names = [f["name"] for f in out]
        self.assertIn("Adenylate-forming coupling fold", names)
        self.assertNotIn("Phosphate-grasp coupling fold", names)

    def test_family_is_covered_once_its_token_is_searched(self):
        families = [self._fam("Adenylate-forming coupling fold", ["adenylate-forming"])]
        self.assertEqual(
            unexplored_promising_families(families, ["adenylate-forming synthetase"]),
            [],
        )

    def test_low_promise_families_are_ignored(self):
        families = [self._fam("Some coupling fold", ["obscuretoken"], promise="low")]
        # Never searched, but low promise → not a coverage requirement.
        self.assertEqual(unexplored_promising_families(families, []), [])

    def test_generic_tokens_do_not_falsely_mark_covered(self):
        # A family described only by generic reaction words is not counted as a
        # requirement (no distinctive token to search for), so it never stalls.
        families = [self._fam("ATP-dependent amide ligase", ["ATP dependent amide bond"])]
        self.assertEqual(unexplored_promising_families(families, ["unrelated query"]), [])

    def test_empty_inputs_are_safe(self):
        self.assertEqual(unexplored_promising_families([], ["x"]), [])
        self.assertEqual(unexplored_promising_families(None, []), [])


if __name__ == "__main__":
    unittest.main()
