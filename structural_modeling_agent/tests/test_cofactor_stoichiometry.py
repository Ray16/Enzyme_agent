"""Deterministic tests for data-driven catalytic-metal stoichiometry (no network)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from cofactor_stoichiometry import metal_count_from_uniprot_json, derive_metal_cofactor


def cofactor_note(text, metal="Mg(2+)"):
    return {"comments": [{"commentType": "COFACTOR",
                          "cofactors": [{"name": metal}],
                          "note": {"texts": [{"value": text}]}}]}


def binding(labels, metal="Mg(2+)"):
    return {"features": [{"type": "Binding site",
                          "ligand": {"name": metal, "label": lab}} for lab in labels]}


class ParserTests(unittest.TestCase):
    def test_explicit_note_two(self):
        c, basis, m = metal_count_from_uniprot_json(cofactor_note("Binds 2 Mg(2+) ions per subunit."))
        self.assertEqual((c, m), (2, "MG"))
        self.assertEqual(basis, "uniprot_cofactor_note")

    def test_explicit_note_one(self):
        c, _, m = metal_count_from_uniprot_json(cofactor_note("Binds 1 Mg(2+) ion per subunit."))
        self.assertEqual((c, m), (1, "MG"))

    def test_binding_labels_two_distinct_ions(self):
        c, basis, m = metal_count_from_uniprot_json(binding(["1", "1", "2", "2"]))
        self.assertEqual(c, 2)
        self.assertEqual(basis, "uniprot_binding_labels")

    def test_binding_chebi_label_is_single_ion(self):
        # several residues binding one unlabelled ion (ChEBI-id label) => 1
        c, _, _ = metal_count_from_uniprot_json(binding(["ChEBI:CHEBI:18420"] * 3))
        self.assertEqual(c, 1)

    def test_cofactor_only_assumes_one(self):
        c, basis, m = metal_count_from_uniprot_json(
            {"comments": [{"commentType": "COFACTOR", "cofactors": [{"name": "Mg(2+)"}]}]})
        self.assertEqual((c, m, basis), (1, "MG", "uniprot_cofactor_single"))

    def test_no_metal_annotation(self):
        c, basis, m = metal_count_from_uniprot_json({"comments": [], "features": []})
        self.assertEqual((c, m, basis), (None, None, "no_metal_annotation"))

    def test_manganese_metal(self):
        c, _, m = metal_count_from_uniprot_json(cofactor_note("Binds 1 Mn(2+) ion.", metal="Mn(2+)"))
        self.assertEqual((c, m), (1, "MN"))


class FallbackTests(unittest.TestCase):
    def test_family_fallback_uses_reviewed_relatives(self):
        """Unannotated candidate -> modal metal count of its InterPro relatives."""
        candidate = {"uniProtKBCrossReferences": [
            {"database": "InterPro", "id": "IPR000873"}]}
        relatives = {  # what the mocked search returns for that family
            "REL1": cofactor_note("Binds 1 Mg(2+) ion per subunit."),
            "REL2": {"comments": [{"commentType": "COFACTOR", "cofactors": [{"name": "Mg(2+)"}]}]},
            "REL3": cofactor_note("Binds 1 Mg(2+) ion per subunit."),
        }

        def fake_fetch(url):
            if url.endswith("CAND.json"):
                return candidate
            for acc, doc in relatives.items():
                if url.endswith(f"{acc}.json"):
                    return doc
            if "search?" in url:  # not used: we monkeypatch member listing below
                return {"results": [{"primaryAccession": a} for a in relatives]}
            return None

        import cofactor_stoichiometry as cs
        orig = cs._reviewed_family_members
        cs._reviewed_family_members = lambda q, limit=25: list(relatives)
        try:
            out = derive_metal_cofactor("CAND", fetch=fake_fetch)
        finally:
            cs._reviewed_family_members = orig
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["ccd"], "MG")
        self.assertTrue(out["basis"].startswith("family_reviewed_modal"))

    def test_generic_default_when_nothing(self):
        out = derive_metal_cofactor("X", fetch=lambda url: None, default_count=1)
        self.assertEqual((out["count"], out["basis"]), (1, "generic_default"))


if __name__ == "__main__":
    unittest.main()
