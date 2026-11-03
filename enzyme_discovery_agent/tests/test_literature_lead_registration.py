import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.literature_leads import register_literature_lead, sequence_name_leads
from enzyme_discovery_agent.models.rolling import evaluation_ready
from enzyme_discovery_agent.models.recovery import next_recovery_action
from enzyme_discovery_agent.tools.reasoning import evaluate_candidate
from enzyme_discovery_agent.register_literature_leads import backfill


class LeadRegistrationTest(unittest.TestCase):
    def prepare(self, root):
        (root / "agent_state").mkdir(exist_ok=True)
        (root / "manifest.json").write_text(json.dumps({"run_id": "original", "status": "budget_exhausted"}))
        text = root / "paper.txt"; text.write_text("Synthetic ZyxQ was expressed and characterized in this study.")
        return text

    def test_unresolved_name_is_visible_idempotent_and_not_evaluated_as_an_accession(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); text = self.prepare(root)
            paper = {"doi": "10.test/primary", "title": "Synthetic reaction study"}
            lead = {"name": "ZyxQ", "organism": "not specified", "relationship": "named"}
            key = register_literature_lead(root, lead, paper, text)
            again = register_literature_lead(root, lead, paper, text)
            self.assertEqual(key, again)
            registry = load_candidate_registry(root)
            self.assertEqual(len(registry["candidates"]), 1)
            entry = registry["candidates"][key]
            self.assertEqual(entry["enzyme_name"], "ZyxQ")
            self.assertEqual(entry["statuses"][-1]["status"], "evidence_pending")
            self.assertEqual(len(entry["statuses"]), 1)
            self.assertFalse(evaluation_ready(entry))
            self.assertTrue(entry["metadata"]["literature_lead"]["name_anchor_found"])
            with patch("enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry") as network:
                response = evaluate_candidate(root, key)
            network.assert_not_called(); self.assertEqual(response["status"], "partial")
            action = next_recovery_action(root, {"search_pubmed"})
            self.assertEqual(action["selection"], "identity_followup")
            self.assertIn("ZyxQ", action["args"]["query"])

    def test_same_symbol_in_different_papers_is_not_a_claim_of_sequence_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); text = self.prepare(root)
            a = register_literature_lead(root, {"name": "ZyxQ"}, {"doi": "10.test/one"}, text)
            b = register_literature_lead(root, {"name": "ZyxQ"}, {"doi": "10.test/two"}, text)
            self.assertNotEqual(a, b)

    def test_bracket_citations_and_line_numbers_do_not_hide_deposited_identifiers(self):
        text = "Genes encoding ZyxQ[12](GenBank: ABC12345.1) and ZyxR[13](GenBank: 332 DEF67890.2) were synthesized."
        leads = sequence_name_leads(text)
        self.assertEqual([(r["name"], r["genbank"]) for r in leads], [("ZyxQ", "ABC12345.1"), ("ZyxR", "DEF67890.2")])

    def test_followup_registers_saved_leads_with_parent_run_and_backup(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d); root = project / "original"; root.mkdir(); text = self.prepare(root)
            register_literature_lead(root, {"name": "ZyxR"}, {"doi": "10.test/old"}, text)
            dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            (dest / "source.txt").write_text("ZyxQ is characterized in this paper.")
            (dest / "manifest.json").write_text(json.dumps([{"doi": "10.test/new", "text_file": "source.txt"}]))
            (dest / "papers_enzymes.json").write_text(json.dumps([{"doi": "10.test/new", "enzymes": [{"name": "ZyxQ", "accessions": []}]}]))
            output = project / "followup"
            result = backfill(root, output, ["10.test/new"])
            self.assertEqual(result["parent_discovery_run_id"], "original")
            self.assertEqual(len(result["newly_registered_lead_ids"]), 1)
            self.assertTrue((output / "before_candidate_registry.json").is_file())
            entry = load_candidate_registry(root)["candidates"][result["newly_registered_lead_ids"][0]]
            self.assertEqual(entry["metadata"]["registration_run_id"], "followup")
            self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "budget_exhausted")


if __name__ == "__main__": unittest.main()


def test_class_labels_do_not_become_candidates_but_named_enzymes_do():
    """A group name can never resolve to a sequence, so it must not occupy a pool slot."""
    from enzyme_discovery_agent.models.literature_leads import enzyme_class_label
    for label in ('NRPS adenylation domains', 'Lipases (ester aminolysis)', 'Penicillin acylases (PACs)',
                  'Amide ligases from Winn et al. 2021', 'Adenylation domain of bacillibactin synthetase',
                  'Coenzyme A ligases and N-acyltransferases (Lovelock library)', 'Lipase', 'Acyltransferase'):
        assert enzyme_class_label(label), label
    for name in ('ShABS', 'CouL', 'XimA', 'AjPPK2-II', 'MsACT acylase', 'McbA',
                 '4-chlorobenzoyl-CoA ligase (4CBCL)', 'MurE amide ligase',
                 'Novobiocic acid synthetase', 'Coronafacate ligase', 'Inorganic pyrophosphatase (IPase)'):
        assert not enzyme_class_label(name), name


def test_a_class_label_carrying_a_deposited_identifier_is_still_registered(tmp_path):
    """The identifier makes the lookup determinate however the paper worded the name."""
    from enzyme_discovery_agent.models.literature_leads import register_literature_lead
    text = tmp_path / 'paper.txt'
    text.write_text('Lipases (ester aminolysis) were cloned and expressed (GenBank: WP_073481158.1).')
    paper = {'doi': '10.1000/test', 'title': 'Test'}
    assert register_literature_lead(tmp_path, {'name': 'Lipases (ester aminolysis)'}, paper, text) is None
    assert register_literature_lead(
        tmp_path, {'name': 'Lipases (ester aminolysis)', 'genbank': 'WP_073481158.1'}, paper, text) is not None
