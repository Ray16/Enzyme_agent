"""Relevance-based derived-family -> InterPro fold resolution.

A derived family must resolve to the surveyable FUNCTIONAL family (whose members
carry the target chemistry), not a coarse structural superfamily matched by a
fold keyword.  The resolver: name-first (authoritative), superfamily/site types
dropped, saturated/singleton entries dropped, broadest coverage first.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.gates.family_relevance import resolve_family_folds
from enzyme_discovery_agent.tools.family_reactions import _LIST_PAGE_MAX


def _entry(acc, typ, name=""):
    return {"accession": acc, "type": typ, "name": name}


class TestResolveFamilyFolds(unittest.TestCase):
    def setUp(self):
        # Fake InterPro search keyed by query term.
        self.db = {
            "functional family": [
                _entry("IPR_FAM", "domain", "the catalytic family"),
                _entry("IPR_SUP", "homologous_superfamily", "a coarse superfamily"),
                _entry("IPR_SITE", "conserved_site", "a conserved site"),
                _entry("IPR_SMALL", "family", "a narrow subfamily"),
            ],
            "structural fold": [  # a noisy descriptor -> unrelated high-count domain
                _entry("IPR_NOISE", "domain", "an unrelated common domain"),
            ],
        }
        # Pre-populated so no network member-count lookup is needed.
        self.count_cache = {
            "IPR_FAM": 188, "IPR_SUP": _LIST_PAGE_MAX, "IPR_SITE": 35,
            "IPR_SMALL": 4, "IPR_NOISE": 400,
        }

    def _search(self, term, max_results=8):
        return {"data": self.db.get(term, [])}

    def test_name_first_superfamily_and_site_dropped(self):
        fam = {"name": "functional family", "descriptors": ["structural fold"]}
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=3)
        # The catalytic family and the narrow subfamily survive; superfamily and
        # conserved_site are dropped by type.
        self.assertIn("IPR_FAM", folds)
        self.assertIn("IPR_SMALL", folds)
        self.assertNotIn("IPR_SUP", folds)
        self.assertNotIn("IPR_SITE", folds)
        # Broadest coverage first.
        self.assertEqual(folds[0], "IPR_FAM")

    def test_descriptor_noise_does_not_outrank_name_entries(self):
        # The noisy descriptor's entry has a higher count (400) than the family
        # (188) but must NOT crowd out the name-derived family: name entries come
        # first, descriptors only fill remaining slots.
        fam = {"name": "functional family", "descriptors": ["structural fold"]}
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=2)
        self.assertEqual(folds, ["IPR_FAM", "IPR_SMALL"])
        self.assertNotIn("IPR_NOISE", folds)

    def test_descriptors_used_when_name_resolves_too_few(self):
        fam = {"name": "unknown term", "descriptors": ["structural fold"]}
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=3)
        self.assertEqual(folds, ["IPR_NOISE"])  # fallback to descriptor

    def test_saturated_and_singleton_excluded(self):
        self.count_cache["IPR_FAM"] = _LIST_PAGE_MAX  # now saturated
        self.count_cache["IPR_SMALL"] = 1             # singleton
        fam = {"name": "functional family", "descriptors": []}
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=3)
        self.assertEqual(folds, [])


if __name__ == "__main__":
    unittest.main()
