"""Tests for the canonical event stream, family key, and search-novelty accounting."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import (
    build_families_index,
    extract_and_register_from_search,
    family_key,
    family_label,
    load_families,
    update_candidate_registry,
)
from enzyme_discovery_agent.models.event_log import append_event, read_events


class TestEventLog(unittest.TestCase):
    def test_append_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_event(root, "run_start", query="q")
            append_event(root, "tool_call", tool="search_uniprot", count=5)
            events = read_events(root)
            self.assertEqual([e["type"] for e in events], ["run_start", "tool_call"])
            # Every event carries the envelope fields.
            for e in events:
                self.assertIn("ts", e)
                self.assertIn("seq", e)
            self.assertEqual(events[0]["query"], "q")
            self.assertEqual(events[1]["count"], 5)

    def test_seq_is_monotonic_and_survives_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_event(root, "a")
            append_event(root, "b")
            seqs = [e["seq"] for e in read_events(root)]
            self.assertEqual(seqs, sorted(seqs))
            self.assertEqual(len(set(seqs)), len(seqs))  # unique

    def test_malformed_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.jsonl").write_text(
                json.dumps({"type": "ok", "seq": 1}) + "\n{ not json\n"
            )
            self.assertEqual([e["type"] for e in read_events(root)], ["ok"])


class TestFamilyKey(unittest.TestCase):
    def test_gene_dominates_and_is_lowercased(self):
        self.assertEqual(family_key({"gene_names": ["MurE"]}, "P1"), "gene:mure")

    def test_falls_back_to_number_stripped_name(self):
        self.assertEqual(
            family_key({"protein_name": "Asparagine synthetase B 2"}, "P1"),
            "name:asparagine synthetase b",
        )

    def test_orthologs_of_one_gene_share_a_key(self):
        # The whole point: different accessions of the same gene collapse.
        k1 = family_key({"gene_names": ["murE"]}, "P22188")
        k2 = family_key({"gene_names": ["murE"]}, "O05954")
        self.assertEqual(k1, k2)


class TestSearchNovelty(unittest.TestCase):
    def _uniprot_result(self, entries):
        return {"status": "success", "summary": "search", "data": entries}

    def test_ortholog_flood_is_capped_to_family_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 4 accessions, all the same gene (murE) = one family, 4 orthologs.
            # The family cap (2) keeps only the best representatives and drops the
            # surplus so discovery budget is not spent re-importing one enzyme.
            entries = [
                {"accession": acc, "gene_names": ["murE"], "name": "Mur ligase",
                 "ec_numbers": [], "seq_length": 450}
                for acc in ("P22188", "O05954", "P57316", "Q03523")
            ]
            result = self._uniprot_result(entries)
            extract_and_register_from_search(root, "search_uniprot", result)

            nov = result["novelty"]
            self.assertEqual(nov["registered"], 2)          # capped at the quota
            self.assertEqual(nov["dropped_orthologs"], 2)   # surplus dropped
            self.assertEqual(nov["distinct_families"], 1)
            self.assertEqual(nov["new_families"], 1)
            # The honest signal is surfaced to the agent, not just the raw count.
            self.assertIn("1 distinct family", result["summary"])
            self.assertIn("dropped", result["summary"])
            # And a structured event was recorded.
            reg = [e for e in read_events(root) if e["type"] == "candidates_registered"]
            self.assertEqual(len(reg), 1)
            self.assertEqual(reg[0]["dropped_orthologs"], 2)

    def test_second_search_sees_family_as_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._uniprot_result(
                [{"accession": "P22188", "gene_names": ["murE"], "name": "n",
                  "ec_numbers": [], "seq_length": 450}]
            )
            extract_and_register_from_search(root, "search_uniprot", first)

            # A different gene AND another murE ortholog: 1 new family, 1 known.
            second = self._uniprot_result([
                {"accession": "P0A6J8", "gene_names": ["ddlA"], "name": "n",
                 "ec_numbers": [], "seq_length": 300},
                {"accession": "O05954", "gene_names": ["murE"], "name": "n",
                 "ec_numbers": [], "seq_length": 450},
            ])
            extract_and_register_from_search(root, "search_uniprot", second)

            nov = second["novelty"]
            self.assertEqual(nov["distinct_families"], 2)   # ddlA + murE
            self.assertEqual(nov["new_families"], 1)        # only ddlA is new
            self.assertEqual(nov["known_accessions"], 0)    # both accessions new
            self.assertEqual(nov["new_accessions"], 2)


class TestFamilyPartition(unittest.TestCase):
    def test_label_prefers_catalytic_interpro_over_gene(self):
        meta = {"gene_names": ["iucC"], "interpro_family_id": "IPR007310",
                "interpro_family": "IucA/IucC"}
        self.assertEqual(family_label(meta, "P1"), "ipr:IPR007310")

    def test_label_falls_back_to_gene_when_unevaluated(self):
        self.assertEqual(family_label({"gene_names": ["murE"]}, "P1"), "gene:mure")

    def test_index_partitions_and_sharpens_display(self):
        candidates = {
            "P1": {"metadata": {"gene_names": ["iucC"],
                                "interpro_family_id": "IPR007310",
                                "interpro_family": "IucA/IucC"},
                   "statuses": [{"status": "discovered"}, {"status": "screened"},
                                {"status": "evaluated"}]},
            "P2": {"metadata": {"gene_names": ["murE"]},
                   "statuses": [{"status": "discovered"}]},
            "P3": {"metadata": {"gene_names": ["murE"]},
                   "statuses": [{"status": "discovered"}]},
        }
        idx = build_families_index(candidates)
        # Two murE orthologs collapse; the IucC lands under its InterPro label.
        self.assertEqual(idx["gene:mure"]["n_members"], 2)
        self.assertIn("ipr:IPR007310", idx)
        self.assertEqual(idx["ipr:IPR007310"]["family"], "IucA/IucC (IPR007310)")
        self.assertIn("evaluated", idx["ipr:IPR007310"]["statuses"])
        # Largest family first.
        self.assertEqual(next(iter(idx)), "gene:mure")

    def test_index_is_written_into_registry_file_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            update_candidate_registry(root, "P22188", "discovered",
                                      metadata={"gene_names": ["murE"]})
            update_candidate_registry(root, "O05954", "discovered",
                                      metadata={"gene_names": ["murE"]})
            # load_families reads the persisted block (not the fallback path).
            fams = json.loads((root / "candidate_registry.json").read_text())
            self.assertIn("families", fams)
            fams = load_families(root)
            self.assertEqual(fams["gene:mure"]["n_members"], 2)


if __name__ == "__main__":
    unittest.main()
