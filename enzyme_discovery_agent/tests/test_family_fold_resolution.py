"""Relevance-based derived-family -> InterPro fold resolution.

A derived family must resolve to the surveyable FUNCTIONAL family (whose members
carry the target chemistry), not a coarse structural superfamily matched by a
fold keyword.  The resolver: name-first (authoritative), superfamily/site types
dropped, saturated/singleton entries dropped, broadest coverage first.
"""

from __future__ import annotations

import unittest
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

from enzyme_discovery_agent.gates.family_relevance import (
    collect_survey_folds,
    resolve_family_fold_candidates,
    resolve_family_folds,
)
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.event_log import read_events
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

    def test_specific_descriptor_can_displace_noisy_name_results(self):
        self.db["noisy family name"] = [
            _entry(f"IPR_NOISE_{i}", "family", f"unrelated family {i}")
            for i in range(5)
        ]
        self.db["specific marker"] = [
            _entry("IPR_RIGHT", "family", "specific marker catalytic family")
        ]
        self.count_cache.update({f"IPR_NOISE_{i}": 300 - i for i in range(5)})
        self.count_cache["IPR_RIGHT"] = 20
        folds = resolve_family_folds({
            "name": "noisy family name", "descriptors": ["specific marker"],
        }, self.count_cache, self._search, max_folds=3)
        self.assertEqual(folds[0], "IPR_RIGHT")

    def test_descriptors_used_when_name_resolves_too_few(self):
        fam = {"name": "unknown term", "descriptors": ["structural fold"]}
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=3)
        self.assertEqual(folds, ["IPR_NOISE"])  # fallback to descriptor

    def test_descriptors_are_skipped_after_name_fills_mapping_quota(self):
        self.db["alpha catalyst family"] = [
            _entry(f"IPR_NAME_{i}", "family", f"alpha catalyst family {i}")
            for i in range(3)
        ]
        self.db["expensive fallback"] = [
            _entry("IPR_FALLBACK", "family", "expensive fallback")
        ]
        self.count_cache.update({f"IPR_NAME_{i}": 10 - i for i in range(3)})
        self.count_cache["IPR_FALLBACK"] = 100
        search = mock.Mock(side_effect=self._search)

        folds = resolve_family_folds({
            "name": "alpha catalyst family",
            "descriptors": ["expensive fallback"],
        }, self.count_cache, search, max_folds=3)

        self.assertEqual(folds, ["IPR_NAME_0", "IPR_NAME_1", "IPR_NAME_2"])
        self.assertEqual([call.args[0] for call in search.call_args_list], [
            "alpha catalyst family",
        ])

    def test_lexical_family_match_outranks_larger_unrelated_entry(self):
        self.db["NIS synthetase family"] = [
            _entry("IPR_WRONG", "family", "Quinolinate synthetase A"),
            _entry("IPR_RIGHT", "domain", "IucA IucC siderophore synthetase"),
        ]
        self.count_cache.update({"IPR_WRONG": 300, "IPR_RIGHT": 20})
        fam = {
            "name": "NIS synthetase family",
            "descriptors": ["IucA IucC siderophore synthetase"],
        }
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=2)
        self.assertEqual(folds, ["IPR_RIGHT", "IPR_WRONG"])

    def test_large_coherent_family_is_not_dropped_as_saturated(self):
        self.count_cache["IPR_FAM"] = _LIST_PAGE_MAX
        self.count_cache["IPR_SMALL"] = 1             # singleton
        fam = {"name": "functional family", "descriptors": []}
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=3)
        self.assertEqual(folds, ["IPR_FAM", "IPR_SMALL"])

    def test_unreviewed_only_coherent_family_remains_auditable(self):
        self.count_cache["IPR_FAM"] = 0
        fam = {"name": "functional family", "descriptors": []}
        folds = resolve_family_folds(fam, self.count_cache, self._search, max_folds=3)
        self.assertIn("IPR_FAM", folds)

    def test_mapping_confidence_comes_from_functional_name_not_descriptor(self):
        self.db["ATP-grasp ligase"] = [
            _entry("IPR_NOISY", "family", "ATP-grasp fold, RimK-type"),
        ]
        self.db["RimK"] = self.db["ATP-grasp ligase"]
        self.count_cache["IPR_NOISY"] = 200
        mappings = resolve_family_fold_candidates({
            "name": "ATP-grasp ligase", "descriptors": ["RimK"],
        }, self.count_cache, self._search)
        self.assertFalse(mappings[0]["mapping_validated"])

    def test_literature_grounded_alias_can_validate_mapping(self):
        self.db["evidence family"] = [
            _entry("IPR_RIGHT", "family", "IucA IucC siderophore synthetase"),
        ]
        self.db["IucA IucC siderophore synthetase"] = self.db["evidence family"]
        self.count_cache["IPR_RIGHT"] = 20
        mappings = resolve_family_fold_candidates({
            "name": "evidence family",
            "descriptors": ["IucA IucC siderophore synthetase"],
            "evidence_basis": "retrieved_literature",
        }, self.count_cache, self._search)
        self.assertTrue(mappings[0]["mapping_validated"])

    def test_partial_token_overlap_does_not_validate_mapping(self):
        self.db["omega ligase family"] = [
            _entry("IPR_GENERIC", "domain", "AMP-dependent synthetase ligase domain"),
        ]
        self.count_cache["IPR_GENERIC"] = 200
        mappings = resolve_family_fold_candidates({
            "name": "omega ligase family",
            "required_catalytic_capability": "ATP-dependent amide bond formation",
        }, self.count_cache, self._search)
        self.assertFalse(mappings[0]["mapping_validated"])


class TestSurveyFoldEligibility(unittest.TestCase):
    def test_qualified_literature_seed_adds_exact_functional_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from enzyme_discovery_agent.models.retrieval_leads import record_evidence_lead
            (root / "target_context.json").write_text(json.dumps({
                "query": "amide formation",
                "target_profile": {"reaction": {
                    "class": "amide formation", "direction": "formation",
                }},
            }))
            record_evidence_lead(
                root, "P00001",
                metadata={"literature_name": "XyzG"},
                evidence={"paper": {"doi": "10.1/example"},
                          "relationship": "abstract_named"},
            )
            with mock.patch(
                "enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light",
                return_value={
                    "ec_numbers": ["6.3.1.1"],
                    "catalytic_activities": ["acid + amine + ATP = amide + ADP + phosphate"],
                },
            ), mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._specific_family_ids",
                return_value=["IPR000009"],
            ):
                folds = collect_survey_folds(root, [], {}, {})
            self.assertEqual(folds, [{
                "fold_id": "IPR000009", "source": "literature_seed:P00001",
            }])

    def test_default_mapping_pool_covers_full_family_inventory(self):
        from enzyme_discovery_agent import config

        self.assertGreaterEqual(
            config.FAMILY_MAPPING_WORKERS, config.MAX_MECHANISM_FAMILIES
        )

    def test_raw_search_hit_does_not_open_whole_family_survey(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            update_candidate_registry(root, "P00001", "discovered", metadata={
                "source_tool": "search_uniprot", "gene_names": ["geneA"],
            })
            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._interpro_ids",
                return_value=["IPR000001"],
            ) as resolve:
                folds = collect_survey_folds(root, [], {}, {})
            self.assertEqual(folds, [])
            resolve.assert_not_called()

    def test_screened_hit_does_not_open_broad_family_survey(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            update_candidate_registry(root, "P00001", "discovered", metadata={
                "source_tool": "search_uniprot", "gene_names": ["geneA"],
            })
            update_candidate_registry(root, "P00001", "screened")
            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._interpro_ids",
                return_value=["IPR000001"],
            ), mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._most_relevant_fold",
                return_value="IPR000001",
            ):
                folds = collect_survey_folds(root, [], {}, {})
            self.assertEqual(folds, [])

    def test_substrate_fit_hit_does_not_recursively_open_domain_surveys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            update_candidate_registry(root, "P00001", "discovered", metadata={
                "source_tool": "family_substrate_fit:IPR000001",
                "gene_names": ["geneA"],
            })
            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._interpro_ids",
                return_value=["IPR000001"],
            ), mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._most_relevant_fold",
                return_value="IPR000001",
            ):
                folds = collect_survey_folds(root, [], {}, {})
            self.assertEqual(folds, [])

    def test_literature_seed_does_not_open_unqualified_family_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from enzyme_discovery_agent.models.retrieval_leads import record_evidence_lead
            record_evidence_lead(root, "P00001", metadata={
                "source_tool": "abstract_named:XyzG@10.1/example",
                "literature_name": "XyzG",
                "gene_names": ["xyzG"],
            })
            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._specific_family_ids",
                return_value=["IPR000001", "IPR000002"],
            ) as resolve:
                folds = collect_survey_folds(root, [], {}, {})
            self.assertEqual(folds, [])
            resolve.assert_not_called()

    def test_shared_resolved_fold_retains_every_hypothesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            families = [
                {"name": "first catalytic route", "search_priority": "focus"},
                {"name": "second catalytic route", "search_priority": "focus"},
            ]
            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance.resolve_family_fold_candidates",
                return_value=[{"fold_id": "IPR_SHARED", "mapping_validated": True}],
            ):
                folds = collect_survey_folds(root, families, {}, {})
            self.assertEqual(len(folds), 1)
            self.assertEqual(
                {item["name"] for item in folds[0]["derived_family_hypotheses"]},
                {"first catalytic route", "second catalytic route"},
            )

    def test_one_mapping_per_hypothesis_is_surveyed_despite_initial_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family = {
                "name": "secondary catalytic route",
                "search_priority": "exploration",
            }
            mappings = [
                {"fold_id": "IPR000001", "mapping_validated": False,
                 "mapping_rank": 1},
                {"fold_id": "IPR000002", "mapping_validated": True,
                 "mapping_rank": 2},
                {"fold_id": "IPR000003", "mapping_validated": False,
                 "mapping_rank": 3},
            ]
            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance.resolve_family_fold_candidates",
                return_value=mappings,
            ):
                folds = collect_survey_folds(root, [family], {}, {})
            self.assertEqual(
                [item["fold_id"] for item in folds],
                ["IPR000001"],
            )
            self.assertTrue(all(
                item["derived_family_hypothesis"]["name"] == "secondary catalytic route"
                for item in folds
            ))

    def test_mapping_rank_uses_predicted_chemistry_and_penalizes_accessory_activity(self):
        entries = [
            _entry("IPR_ACCESSORY", "family", "Delta amidotransferase domain"),
            _entry("IPR_CATALYTIC", "family", "Delta free-substrate coupling family"),
        ]
        counts = {"IPR_ACCESSORY": 500, "IPR_CATALYTIC": 12}

        def search(_term, max_results=20):
            return {"data": entries}

        mappings = resolve_family_fold_candidates({
            "name": "delta coupling family",
            "descriptors": ["delta catalytic fold"],
            "substrates_free": True,
            "required_catalytic_capability":
                "couple a free carboxylate donor to a free amine acceptor",
            "proposed_activation_strategy": "nucleotide-coupled bond formation",
            "disqualifying_observations": [
                "amidotransferase activity without direct donor-acceptor coupling",
            ],
        }, counts, search, max_folds=2)

        self.assertEqual(mappings[0]["fold_id"], "IPR_CATALYTIC")
        self.assertTrue(mappings[0]["mapping_validated"])
        self.assertIn(
            "hypothesis_disqualifier_tokens:1",
            mappings[1]["incompatibility_evidence"],
        )

    def test_catalytic_action_conflict_does_not_win_on_shared_substrate_word(self):
        entries = [
            _entry("IPR_TRANSFER", "domain", "Theta amidotransferase domain"),
            _entry("IPR_FORM", "family", "Theta synthetase catalytic family"),
        ]
        counts = {"IPR_TRANSFER": 600, "IPR_FORM": 18}

        def search(_term, max_results=20):
            return {"data": entries}

        mappings = resolve_family_fold_candidates({
            "name": "theta synthetase-like family",
            "required_catalytic_capability": "ATP-dependent bond formation",
        }, counts, search, max_folds=2)

        self.assertEqual(mappings[0]["fold_id"], "IPR_FORM")
        self.assertIn(
            "conflicting_catalytic_action",
            mappings[1]["incompatibility_evidence"],
        )

    def test_populated_coherent_entry_beats_empty_equal_lexical_match(self):
        entries = [
            _entry("IPR_EMPTY", "family", "Kappa ligase family"),
            _entry("IPR_POPULATED", "family", "Kappa ligase family"),
        ]
        counts = {"IPR_EMPTY": 0, "IPR_POPULATED": 9}

        def search(_term, max_results=20):
            return {"data": entries}

        mappings = resolve_family_fold_candidates({
            "name": "kappa ligase family",
        }, counts, search, max_folds=2)

        self.assertEqual(mappings[0]["fold_id"], "IPR_POPULATED")
        self.assertIn("no_reviewed_members", mappings[1]["incompatibility_evidence"])

    def test_secondary_conflicting_activity_penalizes_multifunction_entry(self):
        entries = [
            _entry("IPR_MIXED", "family", "Lambda synthetase reductase"),
            _entry("IPR_COHERENT", "family", "Lambda synthetase"),
        ]
        counts = {"IPR_MIXED": 80, "IPR_COHERENT": 12}

        def search(_term, max_results=20):
            return {"data": entries}

        mappings = resolve_family_fold_candidates({
            "name": "lambda synthetase family",
            "required_catalytic_capability": "couple two substrates by bond formation",
        }, counts, search, max_folds=2)

        self.assertEqual(mappings[0]["fold_id"], "IPR_COHERENT")
        self.assertIn(
            "conflicting_catalytic_action",
            mappings[1]["incompatibility_evidence"],
        )

    def test_compound_transferase_name_is_recognized_as_conflicting_action(self):
        entries = [
            _entry("IPR_REGULATOR", "family", "Sigma-synthetase adenylyltransferase"),
            _entry("IPR_CATALYTIC", "domain", "Sigma synthetase catalytic domain"),
        ]
        counts = {"IPR_REGULATOR": 100, "IPR_CATALYTIC": 20}

        def search(_term, max_results=20):
            return {"data": entries}

        mappings = resolve_family_fold_candidates({
            "name": "sigma synthetase-like ligase",
            "required_catalytic_capability": "bond formation between two substrates",
        }, counts, search, max_folds=2)

        self.assertEqual(mappings[0]["fold_id"], "IPR_CATALYTIC")
        self.assertIn(
            "conflicting_catalytic_action",
            mappings[1]["incompatibility_evidence"],
        )

    def test_descriptor_search_continues_past_contradictory_name_hits(self):
        db = {"lambda synthetase": [
            _entry(f"IPR_BAD_{i}", "family", f"Lambda synthetase reductase {i}")
            for i in range(3)
        ]}
        db["lambda catalytic family"] = [
            _entry("IPR_GOOD", "family", "Lambda catalytic ligase family")
        ]
        counts = {f"IPR_BAD_{i}": 100 for i in range(3)}
        counts["IPR_GOOD"] = 10

        def search(term, max_results=20):
            return {"data": db.get(term, [])}

        mappings = resolve_family_fold_candidates({
            "name": "lambda synthetase",
            "descriptors": ["lambda catalytic family"],
            "required_catalytic_capability": "couple substrates by bond formation",
        }, counts, search, max_folds=3)

        self.assertEqual(mappings[0]["fold_id"], "IPR_GOOD")

    def test_hypotheses_resolve_concurrently_with_deterministic_reduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            families = [{"name": "family alpha"}, {"name": "family beta"}]
            barrier = threading.Barrier(2)
            log_messages = []

            def resolve(family, _count_cache, _search):
                barrier.wait(timeout=1)
                if family["name"] == "family alpha":
                    time.sleep(0.03)  # Complete second, but remain first in output.
                    fold_id = "IPR_ALPHA"
                else:
                    fold_id = "IPR_BETA"
                return [{"fold_id": fold_id, "mapping_validated": True}]

            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance.resolve_family_fold_candidates",
                side_effect=resolve,
            ), mock.patch(
                "enzyme_discovery_agent.gates.family_relevance.config.FAMILY_MAPPING_WORKERS",
                2,
            ):
                folds = collect_survey_folds(
                    root, families, {}, {}, log_fn=log_messages.append
                )

            self.assertEqual(
                [item["fold_id"] for item in folds], ["IPR_ALPHA", "IPR_BETA"]
            )
            events = read_events(root)
            starts = [e for e in events if e["type"] == "family_mapping_started"]
            completions = [
                e for e in events if e["type"] == "family_mapping_completed"
            ]
            self.assertEqual(len(starts), 2)
            self.assertEqual(len(completions), 2)
            self.assertEqual(
                {e["operation_id"] for e in starts},
                {e["operation_id"] for e in completions},
            )
            self.assertEqual(
                [e["family_name"] for e in completions],
                ["family beta", "family alpha"],
            )
            self.assertTrue(any("2 worker(s)" in msg for msg in log_messages))
            self.assertEqual(
                sum("Family mapping" in msg and "complete in" in msg
                    for msg in log_messages),
                2,
            )


class TestResolutionIdempotence(unittest.TestCase):
    """A re-firing survey pass must not re-resolve already-resolved hypotheses.

    InterPro text resolution is not perfectly stable across calls. Re-resolving
    on every survey pass merged freshly returned entries into the ledger, so the
    resolved fold set grew without bound, each new mapping re-opened a pending
    survey, and the map->survey->map loop never reached a terminal, evaluable
    state. Resolution must therefore run exactly once per hypothesis per run.
    """

    def test_resolved_hypothesis_is_not_reresolved_on_later_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            families = [{"name": "family alpha"}, {"name": "family beta"}]
            calls: list[str] = []

            def resolve(family, _count_cache, _search):
                calls.append(family["name"])
                # Deliberately UNSTABLE: a second resolution of the same family
                # would return a different fold. Idempotence must prevent the
                # second call from ever happening.
                suffix = len([c for c in calls if c == family["name"]])
                return [{
                    "fold_id": f"IPR_{family['name'].split()[-1].upper()}_{suffix}",
                    "mapping_validated": True,
                }]

            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance."
                "resolve_family_fold_candidates",
                side_effect=resolve,
            ):
                first = collect_survey_folds(root, families, {}, {})
                second = collect_survey_folds(root, families, {}, {})

            # Each hypothesis resolved exactly once across BOTH passes.
            self.assertEqual(sorted(calls), ["family alpha", "family beta"])
            # And the surveyed fold set is a stable fixed point.
            self.assertEqual(
                [item["fold_id"] for item in first],
                [item["fold_id"] for item in second],
            )

    def test_empty_resolution_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            families = [{"name": "unresolvable route"}]
            calls: list[str] = []

            def resolve(family, _count_cache, _search):
                calls.append(family["name"])
                return []

            with mock.patch(
                "enzyme_discovery_agent.gates.family_relevance."
                "resolve_family_fold_candidates",
                side_effect=resolve,
            ):
                collect_survey_folds(root, families, {}, {})
                collect_survey_folds(root, families, {}, {})

            # A hypothesis that resolved to nothing is not re-attempted every pass
            # (its verdict is recorded once; re-hitting the network is wasteful and
            # invites the same instability that broke convergence).
            self.assertEqual(calls, ["unresolvable route"])


class TestLiteratureSeedCoverage(unittest.TestCase):
    """A literature-named seed's family is surveyed unless its own annotation
    AFFIRMATIVELY contradicts the target chemistry."""

    def _seed(self, root: Path, direction: str = "formation") -> None:
        from enzyme_discovery_agent.models.retrieval_leads import record_evidence_lead
        (root / "target_context.json").write_text(json.dumps({
            "query": "amide formation",
            "target_profile": {"reaction": {
                "class": "amide formation", "direction": direction,
            }},
        }))
        record_evidence_lead(
            root, "P00001",
            metadata={"literature_name": "XyzG"},
            evidence={"paper": {"doi": "10.1/example"},
                      "relationship": "abstract_named"},
        )

    def test_named_seed_without_annotated_reaction_is_still_surveyed(self):
        # A literature-named enzyme whose UniProt record carries no catalytic
        # activity (common for TrEMBL) must not be lost: its exact family is
        # surveyed and the survey's own gate decides promotion.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root)
            with mock.patch(
                "enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light",
                return_value={"ec_numbers": [], "catalytic_activities": []},
            ), mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._specific_family_ids",
                return_value=["IPR000123"],
            ):
                folds = collect_survey_folds(root, [], {}, {})
            self.assertEqual(folds, [{
                "fold_id": "IPR000123", "source": "literature_seed:P00001",
            }])

    def test_opposite_direction_named_seed_is_dropped(self):
        # A seed whose native reaction runs OPPOSITE the target is a grounded
        # contradiction; its family is not surveyed on the seed's strength.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root)
            with mock.patch(
                "enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light",
                return_value={
                    "ec_numbers": ["3.5.1.1"],
                    "catalytic_activities": [
                        "an amide + H2O = a carboxylate + an amine",
                    ],
                },
            ), mock.patch(
                "enzyme_discovery_agent.gates.family_relevance._specific_family_ids",
                return_value=["IPR000123"],
            ) as resolve:
                folds = collect_survey_folds(root, [], {}, {})
            self.assertEqual(folds, [])
            resolve.assert_not_called()


class TestSurveyCoversExplorationFamilies(unittest.TestCase):
    """The survey must examine every derived family, not just the focus subset.

    focus/exploration is an evaluation-PRIORITY split; a relevant free-substrate
    family can be demoted to exploration (e.g. for an incidental "intramolecular"
    mention). Gating the survey on that split drops relevant families before any
    member is examined, which is the coverage regression this guards against.
    """

    def test_derived_families_include_exploration_routes(self):
        from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
        from enzyme_discovery_agent.models.family_ledger import (
            initialize_family_ledger,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize_family_ledger(root, [
                {"name": "focus ligase family", "search_priority": "focus"},
                {"name": "exploration ligase family",
                 "search_priority": "exploration"},
            ])
            names = {
                f.get("name") for f in FamilySurveyGate()._derived_families(root)
            }
            self.assertEqual(
                names, {"focus ligase family", "exploration ligase family"},
            )


if __name__ == "__main__":
    unittest.main()
