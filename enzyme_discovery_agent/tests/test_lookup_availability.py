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


if __name__ == "__main__":
    unittest.main()
