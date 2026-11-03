"""Identity transitions use synthetic sources and identifiers, independent of campaigns."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.literature_leads import register_literature_lead
from enzyme_discovery_agent.models.recovery import next_recovery_action
from enzyme_discovery_agent.models.rolling import evaluation_ready
from enzyme_discovery_agent.tools.identity_resolution import resolve_literature_lead, select_sequence


def entry(accession="P12345", sequence="MACK", organism="Synthetic species"):
    return {"primaryAccession": accession, "entryType": "UniProtKB unreviewed (TrEMBL)",
            "sequence": {"value": sequence}, "genes": [{"geneName": {"value": "ZyxQ"}}],
            "organism": {"scientificName": organism}, "proteinDescription": {}}


class IdentitySelectionTests(unittest.TestCase):
    def test_same_symbol_in_wrong_taxon_does_not_resolve(self):
        result, _ = select_sequence([entry(organism="Different species")], "ZyxQ", "Synthetic species")
        self.assertIsNone(result)

    def test_multiple_sequences_abstain_even_when_names_and_taxa_match(self):
        result, report = select_sequence([entry(), entry("P54321", "MACD")], "ZyxQ", "Synthetic species")
        self.assertIsNone(result); self.assertEqual(report["status"], "ambiguous")

    def test_equivalent_records_can_resolve_one_sequence(self):
        result, report = select_sequence([entry(), entry("P54321")], "ZyxQ", "Synthetic species")
        self.assertEqual(result["primaryAccession"], "P12345")
        self.assertEqual(len(report["matching_accessions"]), 2)

    def test_partial_sequence_inactive_record_and_missing_gene_are_rejected(self):
        inactive = entry(); inactive["inactiveReason"] = {"type": "DELETED"}
        fragment = entry(); fragment["proteinDescription"]["flag"] = "Fragment"
        wrong = entry(); wrong["genes"] = [{"geneName": {"value": "ZyxQ2"}}]
        for value in [inactive, fragment, wrong, entry(sequence="MACX"), entry(sequence="")]:
            with self.subTest(value=value):
                self.assertIsNone(select_sequence([value], "ZyxQ", "Synthetic species")[0])

    def test_deposited_protein_identifier_must_match_exact_version(self):
        value = entry()
        value["uniProtKBCrossReferences"] = [{"database": "EMBL", "id": "AB123456", "properties": [{"key": "ProteinId", "value": "ABC12345.1"}]}]
        self.assertIsNotNone(select_sequence([value], "Other", "Synthetic species", identifier="ABC12345.1")[0])
        self.assertIsNone(select_sequence([value], "Other", "Synthetic species", identifier="ABC12345.2")[0])


class IdentityLifecycleTests(unittest.TestCase):
    def prepare(self, root, text="ZyxQ from Synthetic species was expressed and purified."):
        path = root / "artifacts" / "paper.txt"; path.parent.mkdir(parents=True)
        path.write_text(text)
        lead = register_literature_lead(root, {"name": "ZyxQ"}, {"doi": "10.test/source"}, path)
        return lead, path

    def test_automatic_recovery_has_a_resolution_action_and_links_both_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lead, _ = self.prepare(root)
            action = next_recovery_action(root, {"resolve_literature_lead"})
            self.assertEqual(action["args"]["lead_id"], lead)
            with patch("enzyme_discovery_agent.tools.fulltext._uniprot_search", return_value=[entry()]):
                result = resolve_literature_lead(lead, run_dir=root)
            self.assertEqual(result["status"], "success")
            registry = load_candidate_registry(root)["candidates"]
            self.assertEqual(registry[lead]["metadata"]["resolved_accession"], "P12345")
            self.assertIn(lead, registry["P12345"]["metadata"]["resolved_literature_leads"])
            self.assertFalse(evaluation_ready(registry[lead]))
            self.assertFalse(result["data"]["activity_validated"])
            self.assertTrue(Path(result["data"]["database_record_path"]).is_file())

    def test_conflicting_sources_never_search_a_preferred_taxon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lead, _ = self.prepare(root)
            extra = root / "artifacts" / "other.txt"; extra.write_text("ZyxQ from Different species was purified.")
            with patch("enzyme_discovery_agent.tools.fulltext._uniprot_search") as search:
                result = resolve_literature_lead(lead, [str(extra)], root)
            search.assert_not_called(); self.assertEqual(result["data"]["status"], "conflict")

    def test_missing_context_changed_source_and_external_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lead, path = self.prepare(root, "ZyxQ was purified.")
            with patch("enzyme_discovery_agent.tools.fulltext._uniprot_search") as search:
                result = resolve_literature_lead(lead, run_dir=root)
                self.assertEqual(result["data"]["status"], "pending")
                result = resolve_literature_lead(lead, [__file__], root)
                self.assertEqual(result["status"], "error")
                path.write_text("Changed source")
                self.assertEqual(resolve_literature_lead(lead, run_dir=root)["status"], "error")
            search.assert_not_called()

    def test_service_failure_retains_resumable_original_lead(self):
        from enzyme_discovery_agent.tools.fulltext import LookupUnavailable
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lead, _ = self.prepare(root)
            with patch("enzyme_discovery_agent.tools.fulltext._uniprot_search", side_effect=LookupUnavailable("temporary outage")):
                result = resolve_literature_lead(lead, run_dir=root)
            self.assertEqual(result["data"]["status"], "unavailable")
            self.assertEqual(set(load_candidate_registry(root)["candidates"]), {lead})

    def test_alias_without_original_source_abstains_without_lookup(self):
        from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            update_candidate_registry(root, "P12345", "evidence_pending", {
                "is_literature_lead": True, "literature_name": "ZyxQ"})
            with patch("enzyme_discovery_agent.tools.fulltext._uniprot_search") as search:
                result = resolve_literature_lead("P12345", run_dir=root)
            self.assertEqual(result["status"], "partial")
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
