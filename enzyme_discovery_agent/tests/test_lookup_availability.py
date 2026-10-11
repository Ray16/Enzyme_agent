"""A failed identity lookup must never be recorded as a resolved negative.

"The database says this enzyme does not exist" is a finding; "we could not
reach the database" is a missing measurement.  Conflating them silently deletes
real candidates, and the deletion leaves no trace in the candidate set -- the
lead is just counted as unresolved.  These tests pin the distinction.
"""

import unittest
from unittest.mock import patch

from enzyme_discovery_agent.tools import fulltext as F


class UniProtSearchAvailability(unittest.TestCase):
    def test_transport_failure_raises_rather_than_reporting_no_hits(self):
        with patch.object(F, "request_with_retry", side_effect=TimeoutError("boom")):
            with self.assertRaises(F.LookupUnavailable):
                F._uniprot_search("gene:FAAH")

    def test_unparseable_body_raises_rather_than_reporting_no_hits(self):
        with patch.object(F, "request_with_retry", return_value=b"<html>502</html>"):
            with self.assertRaises(F.LookupUnavailable):
                F._uniprot_search("gene:FAAH")

    def test_completed_query_with_no_match_returns_empty(self):
        with patch.object(F, "request_with_retry", return_value=b'{"results": []}'):
            self.assertEqual(F._uniprot_search("gene:NOSUCHGENE"), [])

    def test_completed_query_returns_hits(self):
        body = b'{"results": [{"primaryAccession": "O00519"}]}'
        with patch.object(F, "request_with_retry", return_value=body):
            self.assertEqual(
                [h["primaryAccession"] for h in F._uniprot_search("gene:FAAH")], ["O00519"])


class EmblResolveAvailability(unittest.TestCase):
    def test_transport_failure_raises(self):
        with patch.object(F, "request_with_retry", side_effect=OSError("down")):
            with self.assertRaises(F.LookupUnavailable):
                F._resolve_embl_accession("OP972509")

    def test_genome_scale_deposit_is_a_real_negative_not_a_failure(self):
        # Too many cross-referenced proteins is a deliberate decision to skip,
        # which must stay a completed lookup rather than becoming retryable.
        many = b'{"results": [' + b",".join(
            b'{"primaryAccession": "P%05d"}' % i for i in range(60)) + b"]}"
        with patch.object(F, "request_with_retry", return_value=many):
            self.assertEqual(F._resolve_embl_accession("CP000001", cap=50), [])


class NamedEnzymeAvailability(unittest.TestCase):
    def test_resolver_propagates_unavailability_instead_of_returning_none(self):
        """None means 'no such enzyme' and ends the lead; an outage must not."""
        with patch.object(F, "_uniprot_search", side_effect=F.LookupUnavailable("down")):
            with self.assertRaises(F.LookupUnavailable):
                F._resolve_named_enzyme("FAAH", "Homo sapiens", direction="cleavage")

    def test_genuine_absence_still_returns_none(self):
        with patch.object(F, "_uniprot_search", return_value=[]):
            self.assertIsNone(
                F._resolve_named_enzyme("NotAnEnzyme", "Homo sapiens", direction="cleavage"))



class DepositedSequenceFallback(unittest.TestCase):
    """An enzyme absent from UniProt but deposited elsewhere must stay reachable.

    Metagenome-derived and newly published enzymes are routinely deposited in
    GenBank only and acquire no UniProt entry, so the name resolver and the
    EMBL cross-reference resolver both miss them.  Before this fallback such an
    enzyme could be named in a paper, found by the harvest, and still never
    become a candidate.
    """

    def _record(self, acc, title, seq, named):
        return {"accession": acc, "title": title, "organism": "uncultured bacterium",
                "sequence": seq, "sequence_length": len(seq), "title_names_query": named}

    def test_resolves_a_deposit_that_uniprot_does_not_have(self):
        rec = self._record("XYZ12345.1", "hydrolase, partial", "MAREPTALAT" * 12, True)
        with patch("enzyme_discovery_agent.tools.database.search_ncbi_protein",
                   return_value={"status": "success", "data": [rec]}):
            got = F._resolve_named_enzyme_ncbi("SomeNamedEnzyme1")
        self.assertEqual(got["accession"], "XYZ12345.1")
        self.assertTrue(got["identity_confirmed"])
        self.assertEqual(len(got["sequence"]), 120)

    def test_prefers_a_record_that_actually_names_the_enzyme(self):
        weak = self._record("AAA00001.1", "hydrolase, partial", "M" * 300, False)
        strong = self._record("BBB00002.1", "Chain A, SomeNamedEnzyme1", "M" * 200, True)
        with patch("enzyme_discovery_agent.tools.database.search_ncbi_protein",
                   return_value={"status": "success", "data": [weak, strong]}):
            got = F._resolve_named_enzyme_ncbi("SomeNamedEnzyme1")
        self.assertEqual(got["accession"], "BBB00002.1")

    def test_index_only_match_is_kept_but_marked_unconfirmed(self):
        """A generic deposit label must not discard a paper-named enzyme."""
        weak = self._record("AAA00001.1", "ligase, partial", "M" * 300, False)
        with patch("enzyme_discovery_agent.tools.database.search_ncbi_protein",
                   return_value={"status": "success", "data": [weak]}):
            got = F._resolve_named_enzyme_ncbi("SomeNamedEnzyme1")
        self.assertEqual(got["accession"], "AAA00001.1")
        self.assertFalse(got["identity_confirmed"])

    def test_too_generic_a_name_is_not_guessed_at(self):
        with patch("enzyme_discovery_agent.tools.database.search_ncbi_protein") as m:
            self.assertIsNone(F._resolve_named_enzyme_ncbi("ab"))
            m.assert_not_called()

    def test_no_hits_returns_none(self):
        with patch("enzyme_discovery_agent.tools.database.search_ncbi_protein",
                   return_value={"status": "success", "data": []}):
            self.assertIsNone(F._resolve_named_enzyme_ncbi("SomeNamedEnzyme1"))

    def test_ncbi_outage_is_unavailability_not_absence(self):
        with patch("enzyme_discovery_agent.tools.database.search_ncbi_protein",
                   side_effect=OSError("ncbi down")):
            with self.assertRaises(F.LookupUnavailable):
                F._resolve_named_enzyme_ncbi("SomeNamedEnzyme1")

if __name__ == "__main__":
    unittest.main()
