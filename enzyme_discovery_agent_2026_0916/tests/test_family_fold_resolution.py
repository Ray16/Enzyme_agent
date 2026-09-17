"""Relevance-based derived-family -> InterPro fold resolution.

A derived family must resolve to the surveyable FUNCTIONAL family (whose members
carry the target chemistry), not a coarse structural superfamily matched by a
fold keyword.  The resolver: name-first (authoritative), superfamily/site types
dropped, saturated/singleton entries dropped, broadest coverage first.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.gates.family_relevance import (
    _distinctive_family_queries,
    resolve_family_folds,
)
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


class TestDistinctiveFamilyQueries(unittest.TestCase):
    """The query miner must surface the discriminating tokens (gene names,
    enzyme-name bigrams) and drop generic-modifier phrases that carry no signal.
    All abstract/placeholder vocabulary — the miner must name no real enzyme."""

    def test_gene_name_tokens_and_slash_pairs(self):
        fam = {
            "name": "Nax (widget) synthetase fold amide ligase",
            "descriptors": ["Nax synthetase domain (AbcD/AbcE-like)"],
            "literature_terms": ["AbcD/AbcE ligase acting on a dicarboxylate"],
        }
        qs = {q.lower() for q in _distinctive_family_queries(fam)}
        # CamelCase gene tokens and the parts of the slash-joined pair.
        self.assertIn("abcd", qs)
        self.assertIn("abce", qs)

    def test_enzyme_bigram_kept_generic_modifier_dropped(self):
        # "catalytic" is in the shared generic vocabulary; "widget" is not.
        fam = {
            "name": "Widget fold catalytic synthase family",
            "descriptors": ["a distinctive widget synthase domain"],
            "literature_terms": [],
        }
        qs = {q.lower() for q in _distinctive_family_queries(fam)}
        # A distinctive modifier + a -ase head is kept ...
        self.assertIn("widget synthase", qs)
        # ... but "<generic> <head>" carries no discriminating signal and is dropped.
        self.assertNotIn("catalytic synthase", qs)


class TestScoredResolution(unittest.TestCase):
    """The scored path must pick the functionally-relevant fold even when an
    incidental-keyword-match fold has a far higher member count, so a
    non-deterministic verbose family label can no longer lose the on-target fold.
    """

    def setUp(self):
        # On-target entry shares several distinctive tokens with the family;
        # the noise entry shares only a generic word ("synthetase") and has a
        # much higher member count.
        self.db = {
            "AbcD": [_entry("IPR_ON", "domain", "Widget biosynthesis AbcD/AbcE-like")],
            "AbcE": [_entry("IPR_ON", "domain", "Widget biosynthesis AbcD/AbcE-like")],
            "widget synthetase": [
                _entry("IPR_ON", "domain", "Widget biosynthesis AbcD/AbcE-like"),
                _entry("IPR_NOISE", "family", "Polyprenyl synthetase-like"),
            ],
            "Nax synthetase": [_entry("IPR_NOISE", "family", "Polyprenyl synthetase-like")],
        }
        self.count_cache = {"IPR_ON": 20, "IPR_NOISE": 400}

    def _search(self, term, max_results=8):
        return {"data": self.db.get(term, [])}

    def test_relevance_beats_member_count(self):
        fam = {
            "name": "Nax (widget) synthetase fold amide ligase",
            "descriptors": ["Nax synthetase domain (AbcD/AbcE-like)"],
            "literature_terms": ["AbcD/AbcE ligase acting on a dicarboxylate"],
        }
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=4)
        self.assertIn("IPR_ON", folds)          # on-target recovered ...
        self.assertNotIn("IPR_NOISE", folds)    # ... noise (score 0) dropped
        self.assertEqual(folds[0], "IPR_ON")

    def test_falls_back_when_no_distinctive_token_overlap(self):
        # Abstract names with no token overlap -> scored path empty -> the
        # original name-first behaviour resolves it (regression guard).
        db = {"functional family": [_entry("IPR_FAM", "family", "unlabelled entry")]}
        cache = {"IPR_FAM": 100}
        fam = {"name": "functional family", "descriptors": []}
        folds = resolve_family_folds(fam, cache, lambda t, max_results=8: {"data": db.get(t, [])})
        self.assertEqual(folds, ["IPR_FAM"])


if __name__ == "__main__":
    unittest.main()
