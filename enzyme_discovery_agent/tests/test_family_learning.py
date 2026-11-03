"""Evidence must change search behavior even when the model never sets focus."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry
from enzyme_discovery_agent.models.expansion import select_expansion_seeds
from enzyme_discovery_agent.models.family_learning import build_family_learning, candidate_feedback, refresh_family_learning
from enzyme_discovery_agent.models.recovery import next_recovery_action, record_recovery_attempt
from enzyme_discovery_agent.models.rolling import record_evaluation_batch, new_evaluation_work
from enzyme_discovery_agent.gates.family_relevance import allocate_evaluation_budget
from enzyme_discovery_agent.context import build_research_state

TARGET = ("formation", {"amide"}, "free_small_molecule")


def evidence(family, name, fit, reaction="ATP + glutamate + putrescine = ADP + phosphate + glutamylputrescine"):
    return {"interpro_family_id": family, "protein_name": name, "gene_names": [name], "substrate_fit": fit,
            "catalytic_activities": [reaction],
            "reaction_chemistry": {"reaction_direction": "formation", "bond_types": ["amide"]}}


class FamilyLearningTest(unittest.TestCase):
    def prepare(self, root):
        (root / "target_reasoning.json").write_text(json.dumps({
            "reaction_direction": "formation", "bond_types": ["amide"], "substrate_topology": "free_small_molecule"}))
        update_candidate_registry(root, "P12345", "evaluated", evidence("IPR000001", "Protein enzyme", .99,
            "ATP + [protein]-L-lysine + biotin = AMP + biotinyl-[protein]-L-lysine"))
        update_candidate_registry(root, "Q12345", "evaluated", evidence("IPR000002", "Free acid amine ligase", .7))

    def test_supported_small_molecule_native_chemistry_beats_famous_protein_ligase(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root)
            # Even a complete older report putting the protein enzyme first
            # must not freeze the controller after chemistry feedback arrives.
            (root / "pass1_ranking.json").write_text(json.dumps([
                {"accession": "P12345", "rank": 1}, {"accession": "Q12345", "rank": 2}]))
            self.assertEqual(select_expansion_seeds(load_candidate_registry(root)["candidates"], 1, root), ["Q12345"])
            policy = refresh_family_learning(root)
            self.assertEqual(policy["candidates"]["P12345"]["state"], "deprioritized")
            self.assertEqual(policy["priority_order"][0], "ipr:IPR000002")

    def test_unknown_is_not_rejected_and_a_favorable_exception_reopens_family(self):
        unknown = candidate_feedback({"protein_name": "Unknown", "is_enzyme": True}, TARGET)
        self.assertEqual(unknown["state"], "unresolved")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root)
            first = refresh_family_learning(root)
            new = evidence("IPR000001", "New free-substrate exception", .9)
            update_candidate_registry(root, "Q54321", "evaluated", new)
            revised = refresh_family_learning(root)
            self.assertGreater(revised["revision"], first["revision"])
            self.assertEqual(revised["families"]["ipr:IPR000001"]["state"], "supported")
            self.assertEqual(select_expansion_seeds(load_candidate_registry(root)["candidates"], 1, root), ["Q54321"])

    def test_identical_reads_do_not_create_progress_and_homolog_counts_do_not_inflate_support(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root)
            first = refresh_family_learning(root)
            self.assertEqual(refresh_family_learning(root)["revision"], first["revision"])
            for i in range(20):
                update_candidate_registry(root, f"Q123{i:02d}", "evaluated", evidence("IPR000002", "Free acid amine ligase", .7))
            second = refresh_family_learning(root)
            self.assertEqual(second["families"]["ipr:IPR000002"]["independent_enzymes_assessed"], 1)
            self.assertEqual(second["families"]["ipr:IPR000002"]["supported"], 1)

    def test_a_strong_exception_is_not_hidden_by_weak_members_of_a_broad_family(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root)
            update_candidate_registry(root, "Q54321", "evaluated", evidence("IPR000001", "Exceptional member", .9))
            for i in range(5):
                update_candidate_registry(root, f"Q234{i:02d}", "evaluated", evidence("IPR000001", f"Weak member {i}", .1))
            self.assertEqual(select_expansion_seeds(load_candidate_registry(root)["candidates"], 1, root), ["Q54321"])

    def test_budget_moves_to_supported_families_but_preserves_unknown_breadth(self):
        folds = [{"fold_id": fid, "n_members": 10, "dir_rank": 0, "fg_max": fit}
                 for fid, fit in [("good", .5), ("unknown", .9), ("protein", 1.0)]]
        allocation = allocate_evaluation_budget(folds, 10, 1, 6,
            learning_priority={"good": (0, -.7), "unknown": (1, 0), "protein": (2, 0)})
        self.assertEqual(allocation, {"good": 6, "unknown": 3, "protein": 1})

    def test_recovery_follows_evidence_and_reserves_breadth_across_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root)
            (root / "mechanism_families.json").write_text(json.dumps([{
                "name": "Untested mechanism", "promise": "high", "literature_terms": ["distinct unresolved hypothesis"]}]))
            with patch("enzyme_discovery_agent.models.literature_strategy.target_routes", return_value=[]):
                action = next_recovery_action(root, {"search_pubmed"})
            self.assertEqual(action["args"]["query"], "Free acid amine ligase")
            self.assertEqual(action["selection"], "evidence_followup")
            for i in range(4):
                record_recovery_attempt(root, {"key": f"old{i}", "tool": "search_pubmed", "args": {"query": str(i)}}, "empty")
            with patch("enzyme_discovery_agent.models.literature_strategy.target_routes", return_value=[]):
                next_action = next_recovery_action(root, {"search_pubmed"})
            self.assertEqual(next_action["selection"], "exploration")

    def test_updated_family_feedback_reconsiders_deferred_work_without_new_accessions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root)
            update_candidate_registry(root, "Q33333", "discovered", {"protein_name": "Pending enzyme"})
            record_evaluation_batch(root)
            self.assertFalse(new_evaluation_work(root))
            update_candidate_registry(root, "P12345", None, evidence("IPR000001", "Revised enzyme", .9))
            self.assertTrue(new_evaluation_work(root))

    def test_compact_context_keeps_family_comparisons_and_negative_feedback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root)
            state = build_research_state(root)
            rows = state["learned_family_priorities"]
            self.assertEqual(rows[0]["state"], "supported")
            self.assertEqual(rows[0]["median_supported_fit"], .7)
            self.assertIn("ipr:IPR000001", [r["family"] for r in state["families_to_set_aside"]])

    def test_search_yield_counts_distinct_supported_enzymes_not_raw_hits(self):
        candidates = {"P12345": {"statuses": [{"status": "evaluated"}],
                                  "metadata": evidence("IPR000002", "Seed", .7)}}
        for i in range(10):
            meta = evidence("IPR000002", "Same returned enzyme", .7)
            meta["structural_homology"] = {"seed_accession": "P12345"}
            candidates[f"Q123{i:02d}"] = {"statuses": [{"status": "evaluated"}], "metadata": meta}
        policy = build_family_learning(candidates, [], {}, TARGET, {"foldseek": {"P12345": {"count": 10}}})
        self.assertEqual(policy["families"]["ipr:IPR000002"]["new_supported_groups"], 1)
        self.assertEqual(policy["families"]["ipr:IPR000002"]["expansion_attempts"], 1)
        for entry in candidates.values():
            entry["metadata"]["gene_names"] = ["Seed"]
        policy = build_family_learning(candidates, [], {}, TARGET, {"foldseek": {"P12345": {"count": 10}}})
        self.assertEqual(policy["families"]["ipr:IPR000002"]["new_supported_groups"], 0)

    def test_all_surveyed_families_survive_the_bounded_display(self):
        from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
        from enzyme_discovery_agent.gates import AgentState
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gate = FamilySurveyGate()
            folds = [{"fold_id": f"IPR{i:06d}", "functional_group_similarity": {"max": i / 20}}
                     for i in range(14)]
            gate._all_folds = {row["fold_id"]: row for row in folds}
            gate._surveyed_folds = set(gate._all_folds) - {folds[-1]["fold_id"]}
            with patch("enzyme_discovery_agent.gates.family_relevance.collect_survey_folds", return_value=folds), \
                 patch("enzyme_discovery_agent.gates.family_survey.execute_tool", return_value={
                     "status": "success", "data": {"fold_id": folds[-1]["fold_id"], "members": []}}):
                gate.execute(AgentState(root))
            persisted = json.loads((root / "family_landscape.json").read_text())
            self.assertEqual(len(persisted), 14)

    def test_notebook_claims_and_model_confidence_cannot_change_policy(self):
        data = evidence("IPR000002", "Seed", .7)
        a = candidate_feedback(data, TARGET)
        data.update(confidence_score=1.0, reasoning="Definitely catalyzes the target", literature={"characterization_papers": 999})
        self.assertEqual(candidate_feedback(data, TARGET), a)


if __name__ == "__main__":
    unittest.main()
