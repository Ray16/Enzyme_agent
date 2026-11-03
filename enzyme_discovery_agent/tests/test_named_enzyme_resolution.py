"""Direction-gated resolution of an enzyme NAMED in a paper to a UniProt accession.

A characterised enzyme's only DB record is often unreviewed and generically /
mis-annotated (e.g. an amide-bond synthetase recorded as a generic "Fatty acid CoA
ligase"), so the precise fielded name/organism query misses it.  The resolver adds a
free-text organism-anchored fallback that trusts the paper's characterisation over
the stale label — but only for an entry whose enzyme class is consistent with the
target reaction DIRECTION, so a same-symbol unrelated protein is not admitted.
"""

from __future__ import annotations

import unittest
import json
import urllib.parse
from unittest import mock

from enzyme_discovery_agent.tools import fulltext as ft


def _entry(acc, name, ec=None, reviewed=False):
    if reviewed:
        rec = {"fullName": {"value": name}}
        if ec:
            rec["ecNumbers"] = [{"value": ec}]
        pd = {"recommendedName": rec}
        etype = "UniProtKB reviewed (Swiss-Prot)"
    else:
        pd = {"submissionNames": [{"fullName": {"value": name}}]}
        etype = "UniProtKB unreviewed (TrEMBL)"
    return {"primaryAccession": acc, "proteinDescription": pd, "entryType": etype}


class TestDirectionConsistent(unittest.TestCase):
    def test_ligase_is_formation_consistent(self):
        e = _entry("R4R1U5", "Fatty acid CoA ligase")
        self.assertTrue(ft._direction_consistent(e, "formation", "McbA (amide bond synthetase)"))

    def test_hydrolase_not_formation_consistent(self):
        self.assertFalse(ft._direction_consistent(_entry("X", "Amide hydrolase"), "formation", ""))

    def test_ec6_wins_for_formation(self):
        e = _entry("Y", "Some synthetase", ec="6.3.1.2", reviewed=True)
        self.assertTrue(ft._direction_consistent(e, "formation", ""))

    def test_ec3_rejected_for_formation(self):
        e = _entry("Z", "Some hydrolase", ec="3.5.1.4", reviewed=True)
        self.assertFalse(ft._direction_consistent(e, "formation", ""))

    def test_nonenzyme_rejected(self):
        self.assertFalse(ft._direction_consistent(_entry("X", "Uncharacterized membrane protein"),
                                                  "formation", ""))

    def test_unknown_direction_permissive_for_enzyme(self):
        self.assertTrue(ft._direction_consistent(_entry("X", "Some ligase"), None, ""))


class TestResolveNamedEnzyme(unittest.TestCase):
    def test_lookup_requests_valid_return_fields_and_keeps_review_status(self):
        entries = [_entry("Q12345", "Synthetic enzyme", reviewed=False),
                   _entry("P12345", "Synthetic enzyme", reviewed=True)]
        with mock.patch.object(ft, "request_with_retry", return_value=json.dumps({"results": entries}).encode()) as request:
            accession = ft._resolve_named_enzyme("ZyxQ", "Synthetic species")
        fields = urllib.parse.parse_qs(urllib.parse.urlsplit(request.call_args.args[0].full_url).query)["fields"][0].split(",")
        self.assertIn("reviewed", fields)
        self.assertNotIn("entryType", fields)
        self.assertEqual(accession, "P12345")

    def test_organism_constraint_is_never_dropped_to_take_a_symbol_collision(self):
        collision = _entry("P00001", "Some synthetase", reviewed=True)
        collision["genes"] = [{"geneName": {"value": "SynQ"}}]
        def search(query, **kwargs):
            return [] if 'organism_name:' in query else [collision]
        with mock.patch.object(ft, "_uniprot_search", side_effect=search) as query:
            self.assertIsNone(ft._resolve_named_enzyme("SynQ", "Source species"))
        self.assertTrue(all('organism_name:' in call.args[0] for call in query.call_args_list))

    def test_unknown_organism_symbol_does_not_resolve_to_a_noncatalytic_protein(self):
        collision = _entry("P00001", "Membrane scaffold protein", reviewed=True)
        collision["genes"] = [{"geneName": {"value": "SynQ"}}]
        with mock.patch.object(ft, "_uniprot_search", return_value=[collision]) as query:
            self.assertIsNone(ft._resolve_named_enzyme("SynQ", "not specified"))
        self.assertTrue(all('organism_name:' not in call.args[0] for call in query.call_args_list))

    def test_reviewed_gene_match_without_organism_is_not_sequence_identity(self):
        collision = _entry("P00001", "Unrelated synthetic ligase", ec="6.3.1.1", reviewed=True)
        collision["genes"] = [{"geneName": {"value": "SynQ"}}]
        with mock.patch.object(ft, "_uniprot_search", return_value=[collision]) as query:
            self.assertIsNone(ft._resolve_named_enzyme("SynQ", "", direction="formation"))
        query.assert_not_called()

    def test_mcba_resolves_via_freetext_direction_gate(self):
        # Fielded (gene/protein_name) organism query returns nothing; the free-text
        # organism query returns the misannotated unreviewed ligase; the formation
        # direction gate admits it.
        def fake_search(query, size=5, fields="accession"):
            if "gene:" in query or "protein_name:" in query:
                return []  # precise fielded + reviewed-gene queries miss it
            return [_entry("R4R1U5", "Fatty acid CoA ligase")]  # free-text hit
        with mock.patch.object(ft, "_uniprot_search", side_effect=fake_search):
            acc = ft._resolve_named_enzyme(
                "McbA (amide bond synthetase, wild-type)",
                "Marinactinospora thermotolerans", direction="formation")
        self.assertEqual(acc, "R4R1U5")

    def test_wrong_direction_freetext_rejected(self):
        # Same free-text path, but the only hit is a hydrolase — rejected for a
        # formation target, so nothing is registered from a stale-label collision.
        def fake_search(query, size=5, fields="accession"):
            if "gene:" in query or "protein_name:" in query:
                return []
            return [_entry("H1", "Amide hydrolase")]
        with mock.patch.object(ft, "_uniprot_search", side_effect=fake_search):
            acc = ft._resolve_named_enzyme("AmiE (amidohydrolase)", "Some organism",
                                           direction="formation")
        self.assertIsNone(acc)

    def test_precise_fielded_hit_still_preferred(self):
        # When the precise fielded organism query DOES hit, it is used directly
        # (back-compat with the prior precision-first behaviour).
        def fake_search(query, size=5, fields="accession"):
            if "gene:" in query or "protein_name:" in query:
                return [_entry("P00001", "Amide bond synthetase", reviewed=True)]
            return [_entry("WRONG", "unrelated")]
        with mock.patch.object(ft, "_uniprot_search", side_effect=fake_search):
            acc = ft._resolve_named_enzyme("BsnA", "Some organism", direction="formation")
        self.assertEqual(acc, "P00001")


if __name__ == "__main__":
    unittest.main()
