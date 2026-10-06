"""Offline tests for multi-source citation-graph traversal in deep_literature_search.

Stubs the HTTP layer so no network is used; verifies each source's
references(backward)+citations(forward) parsing, cross-source dedup, and that
deep_literature_search merges the extra graphs alongside OpenAlex.
"""

import json
import unittest
from unittest import mock

import enzyme_discovery_agent.tools.literature as L


def _bytes(obj):
    return json.dumps(obj).encode()


class TestSemanticScholarSnowball(unittest.TestCase):
    def test_backward_and_forward_parsed(self):
        def fake_http(req, timeout=30):
            url = req.full_url
            if "/references" in url:
                return _bytes({"data": [{"citedPaper": {
                    "title": "Backward ref", "year": 2019,
                    "externalIds": {"DOI": "10.1/back", "PubMed": "111"},
                    "abstract": "an amidase", "citationCount": 5}}]})
            if "/citations" in url:
                return _bytes({"data": [{"citingPaper": {
                    "title": "Forward citer", "year": 2023,
                    "externalIds": {"DOI": "10.1/fwd"}, "citationCount": 9}}]})
            raise AssertionError(url)
        with mock.patch.object(L, "request_with_retry", fake_http):
            out = L._s2_snowball([{"doi": "10.1/seed", "pmid": "", "pmcid": ""}],
                                 per_seed=10, cap=40)
        dirs = {p["direction"] for p in out}
        self.assertEqual(dirs, {"backward", "forward"})
        self.assertTrue(all(p["source"] == "semanticscholar" for p in out))
        back = [p for p in out if p["direction"] == "backward"][0]
        self.assertEqual(back["doi"], "10.1/back")
        self.assertEqual(back["pmid"], "111")

    def test_no_resolvable_id_skips(self):
        with mock.patch.object(L, "request_with_retry",
                               lambda *a, **k: (_ for _ in ()).throw(AssertionError())):
            self.assertEqual(L._s2_snowball([{"doi": "", "pmid": "", "pmcid": ""}], 10, 40), [])


class TestEuropePMCSnowball(unittest.TestCase):
    def test_references_and_citations_parsed(self):
        def fake_http(req, timeout=30):
            url = req.full_url
            if "/references" in url:
                return _bytes({"referenceList": {"reference": [
                    {"id": "222", "source": "MED", "title": "EPMC back", "pubYear": "2018"}]}})
            if "/citations" in url:
                return _bytes({"citationList": {"citation": [
                    {"id": "333", "source": "MED", "title": "EPMC citer", "pubYear": "2024"}]}})
            raise AssertionError(url)
        with mock.patch.object(L, "request_with_retry", fake_http):
            out = L._europepmc_snowball([{"doi": "", "pmid": "999", "pmcid": ""}], 10, 40)
        self.assertEqual({p["direction"] for p in out}, {"backward", "forward"})
        self.assertTrue(all(p["source"] == "europepmc" for p in out))
        self.assertEqual([p for p in out if p["direction"] == "backward"][0]["pmid"], "222")


class TestPubMedSnowball(unittest.TestCase):
    def test_elink_links_extracted_and_tagged(self):
        def fake_elink(req, timeout=30):
            url = req.full_url
            if "pubmed_pubmed_refs" in url:
                return _bytes({"linksets": [{"linksetdbs": [{"links": ["444", "555"]}]}]})
            if "pubmed_pubmed_citedin" in url:
                return _bytes({"linksets": [{"linksetdbs": [{"links": ["666"]}]}]})
            raise AssertionError(url)
        # stub efetch to avoid depending on PubMed XML parsing internals
        def fake_efetch(pmids):
            return [{"title": f"paper {p}", "abstract": "", "pmid": p} for p in pmids]
        with mock.patch.object(L, "request_with_retry", fake_elink), \
                mock.patch.object(L, "_efetch_pubmed_papers", fake_efetch):
            out = L._pubmed_snowball([{"doi": "", "pmid": "999", "pmcid": ""}], 10, 40)
        by_dir = {p["direction"] for p in out}
        self.assertEqual(by_dir, {"backward", "forward"})
        self.assertTrue(all(p["source"] == "pubmed" for p in out))
        self.assertEqual({p["pmid"] for p in out if p["direction"] == "backward"}, {"444", "555"})

    def test_no_pmid_returns_empty(self):
        self.assertEqual(L._pubmed_snowball([{"doi": "10.1/x", "pmid": "", "pmcid": ""}], 10, 40), [])


class TestCrossSourceDedup(unittest.TestCase):
    def test_same_doi_and_pmid_collapse_across_sources(self):
        papers = [
            {"title": "P", "doi": "10.1/a", "openalex_id": "W1", "source": "openalex"},
            {"title": "P", "doi": "10.1/a", "pmid": "111", "source": "semanticscholar"},  # dup by DOI
            {"title": "Q", "pmid": "222", "source": "europepmc"},
            {"title": "Q other", "pmid": "222", "source": "pubmed"},  # dup by PMID
            {"title": "R", "doi": "10.1/r", "source": "openalex"},
        ]
        out = L._dedup_papers(papers)
        self.assertEqual(len(out), 3)  # {10.1/a}, {222}, {10.1/r}


class TestDeepSearchIntegration(unittest.TestCase):
    def test_extra_sources_merged_and_reported(self):
        seed = {"title": "seed", "doi": "10.1/seed", "pmid": "999", "abstract": "x"}
        ok = {"status": "success", "data": [seed]}
        with mock.patch.object(L, "search_pubmed", lambda *a, **k: ok), \
             mock.patch.object(L, "search_openalex", lambda *a, **k: {"status": "success", "data": []}), \
             mock.patch.object(L, "search_europepmc", lambda *a, **k: {"status": "success", "data": []}), \
             mock.patch.object(L, "search_semantic_scholar", lambda *a, **k: {"status": "success", "data": []}), \
             mock.patch.object(L, "request_with_retry", lambda *a, **k: _bytes({"results": []})), \
             mock.patch.object(L, "_s2_snowball", lambda ids, ps, cap: [
                 {"title": "s2 hit", "doi": "10.9/s2", "abstract": "an amidase EC 3.5.1.4",
                  "source": "semanticscholar", "direction": "forward"}]), \
             mock.patch.object(L, "_europepmc_snowball", lambda ids, ps, cap: []), \
             mock.patch.object(L, "_pubmed_snowball", lambda ids, ps, cap: []):
            res = L.deep_literature_search("find some enzymes", max_seed_papers=1)
        stats = res["data"]["stats"]
        self.assertIn("extra_source_refs", stats)
        self.assertEqual(stats["extra_source_refs"]["semanticscholar"], 1)
        self.assertGreaterEqual(stats["references_fetched"], 1)


if __name__ == "__main__":
    unittest.main()
