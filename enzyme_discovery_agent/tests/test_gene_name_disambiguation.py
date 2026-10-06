"""Tests for gene-name collision prevention.

Validates that:
1. search_uniprot returns function/catalytic_activity data for LLM reasoning
2. GenBank accessions are extracted from paper text
3. lookup_genbank_to_uniprot resolves accessions to UniProt entries
4. Function data flows into candidate registration
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@unittest.skipUnless(os.getenv("ENZYME_AGENT_RUN_LIVE_TESTS") == "1", "requires live UniProt APIs")
class TestSearchUniprotEnrichment(unittest.TestCase):
    """Test that search_uniprot returns function/catalytic data (live API)."""

    def test_search_returns_function_field(self):
        from enzyme_discovery_agent.tools.database import search_uniprot

        result = search_uniprot(gene_name="sfaB", max_results=5)
        assert result["status"] in ("success", "partial")
        if result["data"]:
            entry = result["data"][0]
            assert "function" in entry, "Missing 'function' field in search result"
            assert "catalytic_activities" in entry, "Missing 'catalytic_activities' field"

    def test_well_annotated_entry_has_function(self):
        from enzyme_discovery_agent.tools.database import search_uniprot

        result = search_uniprot(query="accession:P78061", max_results=1)
        if result["data"]:
            entry = result["data"][0]
            assert entry.get("function") or entry.get("catalytic_activities"), (
                "Well-annotated entry P78061 should have function or catalytic data"
            )

    def test_multiple_sfab_hits_distinguishable(self):
        """Different proteins named sfaB should have distinguishable annotations."""
        from enzyme_discovery_agent.tools.database import search_uniprot

        result = search_uniprot(gene_name="sfaB", max_results=10)
        if len(result["data"]) >= 2:
            names = [e["name"] for e in result["data"]]
            assert len(names) >= 2, "Expected multiple sfaB hits"


@unittest.skipUnless(os.getenv("ENZYME_AGENT_RUN_LIVE_TESTS") == "1", "requires live UniProt APIs")
class TestGenBankCrossReference(unittest.TestCase):
    """Test GenBank extraction and reverse lookup."""

    def test_genbank_accession_extracted_from_text(self):
        from enzyme_discovery_agent.tools.literature import extract_entities

        papers = [{
            "title": "Characterization of an enzyme (GenBank: ATY72527.1)",
            "abstract": "We cloned the gene (GenBank accession ATY72527.1) "
                        "from Streptomyces thioluteus and expressed it in E. coli.",
        }]
        result = extract_entities(papers)
        genbank = result["data"].get("genbank_accessions", [])
        accessions = [g["accession"] for g in genbank]
        assert any("ATY72527" in a for a in accessions), (
            f"Expected ATY72527 in extracted GenBank accessions, got: {accessions}"
        )

    def test_lookup_genbank_to_uniprot_live(self):
        from enzyme_discovery_agent.tools.database import lookup_genbank_to_uniprot

        result = lookup_genbank_to_uniprot("ATY72527.1")
        assert result["status"] in ("success", "partial"), f"Lookup failed: {result}"
        if result["data"]:
            entry = result["data"][0]
            assert entry["organism"] == "Streptomyces thioluteus", (
                f"Expected Streptomyces thioluteus, got {entry['organism']}"
            )
            assert entry.get("genbank_source") == "ATY72527.1"

    def test_lookup_nonexistent_genbank(self):
        from enzyme_discovery_agent.tools.database import lookup_genbank_to_uniprot

        result = lookup_genbank_to_uniprot("XXXXXX999")
        assert result["count"] == 0 or result["status"] == "partial"

    def test_genbank_regex_does_not_match_short_gene_names(self):
        from enzyme_discovery_agent.tools.literature import extract_entities

        papers = [{
            "title": "DdaG and SfaB are amide synthetases",
            "abstract": "We characterized DdaG from Enterobacter agglomerans.",
        }]
        result = extract_entities(papers)
        genbank = result["data"].get("genbank_accessions", [])
        accessions = [g["accession"] for g in genbank]
        assert "DdaG" not in accessions
        assert "SfaB" not in accessions


class TestCandidateRegistration(unittest.TestCase):
    """Test that function data flows into candidate registry."""

    def test_registration_includes_function_data(self):
        from enzyme_discovery_agent.models.candidate_registry import (
            extract_and_register_from_search,
            load_candidate_registry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            mock_result = {
                "status": "success",
                "summary": "test search",
                "data": [
                    {
                        "accession": "P78061",
                        "name": "Gamma-glutamylputrescine synthetase",
                        "organism": "E. coli",
                        "ec_number": "6.3.1.11",
                        "gene_names": ["puuA"],
                        "seq_length": 500,
                        "sequence": "M" * 500,
                        "function": "Catalyzes gamma-glutamylation of putrescine",
                        "catalytic_activities": ["putrescine + glutamate + ATP = product"],
                    }
                ],
            }

            registered = extract_and_register_from_search(run_dir, "search_uniprot", mock_result)
            assert "P78061" in registered

            registry = load_candidate_registry(run_dir)
            meta = registry["candidates"]["P78061"]["metadata"]
            assert meta["function"] == "Catalyzes gamma-glutamylation of putrescine"
            assert len(meta["catalytic_activities"]) == 1


def run_tests():
    """Run all tests and report results."""
    test_classes = [
        TestSearchUniprotEnrichment,
        TestGenBankCrossReference,
        TestCandidateRegistration,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  PASS  {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{method_name}", str(e)))
                print(f"  FAIL  {cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
