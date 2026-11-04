import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from campaign_catalog.catalog import build, verify, audit_events, record_feedback
from campaign_catalog.prioritization import review_priority


class CatalogTest(unittest.TestCase):
    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data)); return path

    def csv(self, path, rows):
        out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(out.getvalue()); return path

    def entry(self, accession, status, timestamp):
        return {"accession": accession, "last_updated": timestamp, "statuses": [{"status": status, "timestamp": timestamp}],
                "metadata": {"protein_name": "Synthetic catalyst", "gene_names": ["zyxQ"], "interpro_family_id": "IPR000001"}}

    def fixture(self, project):
        campaign = project / "campaign"; campaign.mkdir()
        sources = []
        for name, entries in [("reference", {"P12345": self.entry("P12345", "evaluated", "2026-01-01T00:00:00+00:00"),
                                             "Q12345": self.entry("Q12345", "evaluated", "2026-01-01T00:00:00+00:00")}),
                              ("current", {"P12345": self.entry("P12345", "rejected", "2026-02-01T00:00:00+00:00"),
                                           "Q54321": self.entry("Q54321", "evaluated", "2026-02-01T00:00:00+00:00")})]:
            root = project / name
            self.put(root / "manifest.json", {"run_id": name, "status": "completed"})
            self.put(root / "candidate_registry.json", {"candidates": entries})
            self.put(root / "target_context.json", {"substrate_smiles": "CCO", "target_reasoning": {"reaction_direction": "redox"}})
            ranking = self.csv(root / "ranking.csv", [{"accession": key, "rank": i + 1} for i, key in enumerate(entries)])
            sources.append({"path": str(root), "label": name, "target_id": "synthetic", "ranking": str(ranking), "baseline": name == "reference"})
        model = project / "models"; predictions = model / "predictions"; predictions.mkdir(parents=True)
        (predictions / "example_model_0.pdb").write_text("synthetic coordinate artifact\n")
        self.put(predictions / "confidence_example_model_0.json", {"confidence": 1})
        self.csv(model / "selected.csv", [{"accession": "P12345"}, {"accession": "Q12345"}])
        self.csv(model / "modeled.csv", [{"accession": "P12345", "source": "reused", "n_models": 1}])
        self.csv(model / "excluded.csv", [{"accession": "Q12345", "exclusion_reason": "Unsupported setup"}])
        self.put(model / "index.json", {"candidates": [{"accession": "P12345", "prediction_dir": str(predictions)}]})
        config = {"campaign_id": "synthetic", "project_root": str(project), "discovery_sources": sources,
                  "modeling_sources": [{"run_id": "collection", "path": str(model), "label": "historical collection",
                      "target_id": "synthetic", "discovery_run_id": "reference", "selection": str(model / "selected.csv"),
                      "modeled_manifest": str(model / "modeled.csv"), "excluded_manifest": str(model / "excluded.csv"),
                      "candidate_index": str(model / "index.json"), "inputs_dir": str(model / "inputs"),
                      "expected": {"selected": 2, "modeled": 1, "excluded": 1},
                      "producer_runs": {"reused": {"run_id": "earlier_producer", "label": "earlier source", "scope": "legacy identifier"}}}]}
        self.put(campaign / "sources.json", config)
        return campaign

    def rows(self, campaign, name):
        pointer = json.loads((campaign / "catalog/current.json").read_text())
        folder = Path(pointer["manifest"]).parent
        return [json.loads(line) for line in (folder / (name + ".jsonl")).read_text().splitlines()]

    def test_shared_identity_retains_per_run_decisions_and_model_producer(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = self.fixture(Path(d)); result = build(campaign)
            self.assertEqual(result["counts"]["proteins"], 3)
            self.assertEqual(result["counts"]["assessments"], 4)
            assessments = [r for r in self.rows(campaign, "assessments") if r["protein_id"] == "uniprot:P12345"]
            self.assertEqual({r["status"] for r in assessments}, {"evaluated", "rejected"})
            model = self.rows(campaign, "modeling_executions")[0]
            self.assertEqual(model["producer_run_id"], "earlier_producer")
            self.assertEqual(model["collection_run_id"], "collection")
            self.assertEqual(model["output_presence"], "complete")
            self.assertEqual(len(model["artifacts"]), 2)
            self.assertTrue(verify(campaign)["snapshot_verified"])
            pool = list(csv.DictReader((campaign / "views/candidate_pool.csv").open()))
            self.assertEqual(next(r for r in pool if r["accession"] == "P12345")["latest_status"], "rejected")
            queue = list(csv.DictReader((campaign / "views/new_modeling_review.csv").open()))
            self.assertEqual([r["accession"] for r in queue], ["Q54321"])

    def test_refresh_is_idempotent_and_changes_create_immutable_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d); campaign = self.fixture(project); initial = build(campaign)
            again = build(campaign); self.assertEqual(initial["snapshot_id"], again["snapshot_id"])
            self.assertFalse(again["new_snapshot"])
            source = project / "current/candidate_registry.json"; data = json.loads(source.read_text())
            data["candidates"]["Q54321"]["metadata"]["protein_name"] = "Revised evidence label"; self.put(source, data)
            changed = build(campaign); self.assertNotEqual(initial["snapshot_id"], changed["snapshot_id"])
            old = campaign / "catalog/snapshots" / initial["snapshot_id"] / "assessments.jsonl"
            self.assertNotIn("Revised evidence label", old.read_text())
            self.assertEqual(len(audit_events(campaign / "catalog/events.jsonl")), 3)

    def test_invalid_partition_and_target_mismatch_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d); campaign = self.fixture(project)
            self.csv(project / "models/excluded.csv", [{"accession": "P12345", "exclusion_reason": "conflict"}])
            with self.assertRaises(ValueError): build(campaign)
            self.assertFalse((campaign / "catalog/current.json").exists())
        with tempfile.TemporaryDirectory() as d:
            project = Path(d); campaign = self.fixture(project)
            self.put(project / "current/target_context.json", {"substrate_smiles": "CCN"})
            with self.assertRaisesRegex(ValueError, "chemistry contracts"): build(campaign)

    def test_tampering_is_detected_and_source_records_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d); campaign = self.fixture(project)
            source = project / "current/candidate_registry.json"; before = source.read_bytes(); build(campaign)
            self.assertEqual(source.read_bytes(), before)
            pointer = json.loads((campaign / "catalog/current.json").read_text()); snap = Path(pointer["manifest"]).parent
            (snap / "proteins.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError): verify(campaign)
            with self.assertRaises(ValueError): build(campaign)

    def test_policy_revision_and_feedback_are_recorded_without_changing_old_assessments(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d); campaign = self.fixture(project)
            policy = self.put(campaign / "policy.json", {"version": "1", "change_reason": "Initial evidence gates"})
            config = json.loads((campaign / "sources.json").read_text()); config["prioritization_policy"] = str(policy)
            self.put(campaign / "sources.json", config); first = build(campaign)
            evidence = self.put(project / "observation.json", {"observation": "Source-linked result"})
            feedback = self.put(project / "feedback.json", {"protein_id": "uniprot:P12345", "source_run_id": "reference",
                "kind": "expression_observation", "result": "unknown", "evidence_files": [str(evidence)]})
            first_feedback = record_feedback(campaign, feedback); record_feedback(campaign, feedback)
            self.assertEqual(len(audit_events(campaign / "catalog/feedback_events.jsonl")), 1)
            self.put(evidence, {"observation": "Updated source observation"})
            self.assertNotEqual(record_feedback(campaign, feedback)["observation_id"], first_feedback["observation_id"])
            self.put(policy, {"version": "2", "change_reason": "Refined after recorded observations"})
            next_build = build(campaign)
            self.assertNotEqual(next_build["snapshot_id"], first["snapshot_id"])
            self.assertEqual(len(self.rows(campaign, "feedback")), 2)
            old = campaign / "catalog/snapshots" / first["snapshot_id"] / "policies.jsonl"
            self.assertEqual(json.loads(old.read_text())["policy"]["version"], "1")
            self.put(feedback, {"protein_id": "uniprot:Q54321", "source_run_id": "reference",
                "kind": "pose_validation", "result": "pass", "evidence_files": [str(evidence)]})
            with self.assertRaisesRegex(ValueError, "does not belong"): record_feedback(campaign, feedback)

    def test_priority_gates_preserve_uncertainty_and_require_an_informative_job(self):
        policy = {"version": "test"}
        base = {"accession": "P12345", "latest_status": "evaluated", "host_evidence": "reported_active"}
        self.assertEqual(review_priority({**base, "previously_modeled": True}, policy)["modeling_priority_state"], "review_existing_models")
        self.assertEqual(review_priority({**base, "assessment_quality": "unresolved source"}, policy)["modeling_priority_state"], "resolve_relevance_and_scoring_evidence")
        self.assertEqual(review_priority({**base, "family_already_modeled": True}, policy)["modeling_priority_state"], "justify_advantage_over_existing_representative")
        decision = review_priority(base, policy)
        self.assertEqual(decision["modeling_priority_state"], "define_modeling_question_and_validate_setup")
        self.assertFalse(decision["ready_for_submission"])

    def test_unresolved_paper_names_are_preserved_without_creating_verified_proteins(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d); campaign = self.fixture(project)
            lead = {"name": "Synthetic ZyxR", "organism": "Synthetic species", "accessions": [], "relationship": "named"}
            self.put(project / "current/artifacts/fulltext/papers_enzymes.json", [{
                "doi": "10.test/primary", "title": "Synthetic reaction study", "enzymes": [lead, lead], "homologs": []}])
            result = build(campaign)
            self.assertEqual(result["counts"]["proteins"], 3)
            leads = self.rows(campaign, "literature_leads")
            self.assertEqual(len(leads), 1)
            self.assertEqual(leads[0]["associated_identifiers"], [])
            self.assertEqual(leads[0]["source_run_id"], "current")
            self.assertEqual(len(leads[0]["sources"]), 2)


if __name__ == "__main__": unittest.main()
