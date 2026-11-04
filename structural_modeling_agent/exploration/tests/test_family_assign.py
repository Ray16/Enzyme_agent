"""Unit tests for family_assign pure parsers + coordinate bridge (no network)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import family_assign as fa  # noqa: E402


class TestInterProScanParser(unittest.TestCase):
    def test_extracts_integrated_interpro_entries(self):
        payload = {"results": [{"matches": [
            {"signature": {"accession": "PF00501", "name": "AMP-binding",
                           "signatureLibraryRelease": {"library": "PFAM"},
                           "entry": {"accession": "IPR000873",
                                     "name": "AMP-dependent synthetase/ligase",
                                     "type": "DOMAIN"}}},
            {"signature": {"accession": "PS00455",
                           "signatureLibraryRelease": {"library": "PROSITE_PATTERNS"},
                           "entry": {"accession": "IPR000873", "name": "AMP-dependent",
                                     "type": "DOMAIN"}}},
        ]}]}
        entries = fa.parse_interproscan_json(payload)
        self.assertEqual(len(entries), 1)  # deduped by InterPro accession
        self.assertEqual(entries[0]["accession"], "IPR000873")
        self.assertEqual(len(entries[0]["signatures"]), 2)

    def test_skips_signatures_without_integrated_entry(self):
        payload = {"results": [{"matches": [
            {"signature": {"accession": "PF99999", "entry": None}},
        ]}]}
        self.assertEqual(fa.parse_interproscan_json(payload), [])

    def test_empty_payload(self):
        self.assertEqual(fa.parse_interproscan_json({}), [])


class TestUniProtParser(unittest.TestCase):
    def test_extracts_interpro_xrefs_only(self):
        entry = {"uniProtKBCrossReferences": [
            {"database": "InterPro", "id": "IPR000873",
             "properties": [{"key": "EntryName", "value": "AMP-dependent"}]},
            {"database": "Pfam", "id": "PF00501"},
        ]}
        out = fa.parse_uniprot_interpro(entry)
        self.assertEqual([x["accession"] for x in out], ["IPR000873"])
        self.assertEqual(out[0]["name"], "AMP-dependent")


class TestFoldseekParser(unittest.TestCase):
    def test_parse_m8_and_accession(self):
        text = "seed\tAF-Q46E79-F1-model_v4\t1e-30\t250.5\nseed\tbadrow\n"
        hits = fa.parse_foldseek_m8(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["bits"], 250.5)
        self.assertEqual(fa.uniprot_from_foldseek_target(hits[0]["target"]), "Q46E79")

    def test_accession_none_on_nonafdb(self):
        self.assertIsNone(fa.uniprot_from_foldseek_target("6xyz_A"))


class TestCoordinateBridge(unittest.TestCase):
    def _coords(self):
        return {"mechanisms": {
            "anl": {"keys": {"families": ["IPR000873", "IPR037455"]},
                    "steps": [{"index": 1}, {"index": 2}]},
            "other": {"keys": {"families": ["IPR999999"]}, "steps": [{"index": 1}]},
        }}

    def test_family_match_selects_coordinate(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(self._coords(), f)
            path = f.name
        try:
            asg = fa.FamilyAssignment(target="MGYP1", family_ids=("IPR037455",))
            rc = fa.coordinate_for(asg, path)
            self.assertEqual(rc["coordinate_key"], "anl")
            self.assertEqual(rc["basis"], "family")
            self.assertEqual(rc["matched_families"], ["IPR037455"])
            self.assertEqual(rc["n_steps"], 2)
        finally:
            os.unlink(path)

    def test_no_family_match_abstains(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(self._coords(), f)
            path = f.name
        try:
            asg = fa.FamilyAssignment(target="MGYP2", family_ids=("IPR123456",))
            rc = fa.coordinate_for(asg, path)
            self.assertIsNone(rc["coordinate_key"])
        finally:
            os.unlink(path)


class TestFamilyAssignment(unittest.TestCase):
    def test_assigned_flag_and_serialization(self):
        asg = fa.FamilyAssignment(target="MGYP", family_ids=("IPR000873",), method="interproscan")
        self.assertTrue(asg.assigned)
        d = asg.to_dict()
        self.assertEqual(d["family_ids"], ["IPR000873"])
        self.assertFalse(fa.FamilyAssignment(target="x").assigned)


if __name__ == "__main__":
    unittest.main()
