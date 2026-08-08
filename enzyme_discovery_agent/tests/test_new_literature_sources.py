"""Tests for the Europe PMC / Semantic Scholar / Crossref retrievers.

These are the first tests to exercise the literature retrievers themselves.
The network layer (``request_with_retry``) is mocked with realistic fixture
payloads so parsing, the standardized return contract, and identifier
extraction are verified without any live calls.  Semantic re-ranking is off by
default (``config.ENABLE_SEMANTIC_RERANK``), so the model is never loaded here.
"""

import json
import unittest
from unittest import mock

from enzyme_discovery_agent.tools import literature

_HTTP = "enzyme_discovery_agent.tools.literature.request_with_retry"


def _bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode()


class TestEuropePMC(unittest.TestCase):
    FIXTURE = {
        "resultList": {"result": [{
            "title": "Directed evolution of an amide synthetase",
            "pubYear": "2023",
            "doi": "10.1000/abc123",
            "pmid": "39000001",
            "pmcid": "PMC1234567",
            "journalTitle": "Nature Catalysis",
            "authorString": "Doe J, Roe A, Smith B, Lee C, Kim D, Park E",
            "abstractText": "We engineered an amide synthetase ...",
            "isOpenAccess": "Y",
            "citedByCount": 42,
            "fullTextUrlList": {"fullTextUrl": [
                {"availability": "Open access", "documentStyle": "pdf",
                 "url": "https://europepmc.org/articles/PMC1234567?pdf=render"},
            ]},
        }]}
    }

    def test_parses_ids_and_oa_pdf(self):
        with mock.patch(_HTTP, return_value=_bytes(self.FIXTURE)):
            out = literature.search_europepmc("amide synthetase", max_results=5)
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["count"], 1)
        p = out["data"][0]
        self.assertEqual(p["pmcid"], "PMC1234567")
        self.assertEqual(p["pmid"], "39000001")
        self.assertEqual(p["year"], 2023)
        self.assertTrue(p["is_open_access"])
        self.assertIn("pdf=render", p["oa_pdf_url"])
        self.assertEqual(len(p["authors"]), 5)  # capped at 5
        self.assertEqual(p["source"], "europepmc")

    def test_oa_pdf_falls_back_to_render_url(self):
        fixture = {"resultList": {"result": [{
            "title": "X", "pubYear": "2022", "pmcid": "PMC9",
            "isOpenAccess": "N",  # boolean lags; render URL still derived
        }]}}
        with mock.patch(_HTTP, return_value=_bytes(fixture)):
            out = literature.search_europepmc("x", max_results=5)
        self.assertEqual(out["data"][0]["oa_pdf_url"],
                         "https://europepmc.org/articles/PMC9?pdf=render")

    def test_empty_results_is_partial(self):
        with mock.patch(_HTTP, return_value=_bytes({"resultList": {"result": []}})):
            out = literature.search_europepmc("nothing", max_results=5)
        self.assertEqual(out["status"], "partial")
        self.assertEqual(out["count"], 0)


class TestSemanticScholar(unittest.TestCase):
    FIXTURE = {"data": [{
        "title": "De novo design of luciferase enzymes",
        "year": 2024,
        "venue": "Nature",
        "abstract": "We designed luciferases ...",
        "externalIds": {"DOI": "10.1000/lux", "PubMed": "40000002",
                        "PubMedCentral": "PMC7654321", "ArXiv": "2401.12345"},
        "openAccessPdf": {"url": "https://repo.example/paper.pdf", "status": "GREEN"},
        "citationCount": 7,
        "authors": [{"name": "A One"}, {"name": "B Two"}],
    }]}

    def test_parses_full_identifier_set(self):
        with mock.patch(_HTTP, return_value=_bytes(self.FIXTURE)):
            out = literature.search_semantic_scholar("luciferase design", max_results=5)
        self.assertEqual(out["status"], "success")
        p = out["data"][0]
        self.assertEqual(p["doi"], "10.1000/lux")
        self.assertEqual(p["pmid"], "40000002")
        self.assertEqual(p["pmcid"], "PMC7654321")
        self.assertEqual(p["arxiv_id"], "2401.12345")
        self.assertEqual(p["oa_status"], "GREEN")
        self.assertEqual(p["oa_pdf_url"], "https://repo.example/paper.pdf")
        self.assertEqual(p["source"], "semantic_scholar")

    def test_rate_limit_returns_actionable_partial(self):
        with mock.patch(_HTTP, side_effect=Exception("HTTP Error 429: Too Many Requests")):
            out = literature.search_semantic_scholar("x", max_results=5)
        self.assertEqual(out["status"], "partial")
        self.assertTrue(any("ENZYME_AGENT_S2_API_KEY" in s for s in out["suggestions"]))

    def test_api_key_header_sent_when_configured(self):
        captured = {}

        def _fake(req, timeout=30):
            captured["headers"] = {k.lower(): v for k, v in dict(req.headers).items()}
            return _bytes(self.FIXTURE)

        with mock.patch.object(literature.config, "S2_API_KEY", "secret-key"), \
                mock.patch(_HTTP, side_effect=_fake):
            literature.search_semantic_scholar("x", max_results=5)
        self.assertEqual(captured["headers"].get("x-api-key"), "secret-key")


class TestCrossref(unittest.TestCase):
    FIXTURE = {"message": {"items": [{
        "DOI": "10.1021/sb500299c",
        "title": ["Directed evolution of orthogonal T7 RNA polymerase variants"],
        "container-title": ["ACS Synthetic Biology"],
        "author": [{"given": "Jane", "family": "Doe"}, {"given": "Rob", "family": "Roe"}],
        "issued": {"date-parts": [[2015, 3]]},
        "is-referenced-by-count": 120,
        "score": 50.9,
        "abstract": "<jats:p>We evolved <jats:italic>T7</jats:italic> variants.</jats:p>",
        "type": "journal-article",
    }]}}

    def test_parses_doi_score_and_strips_jats(self):
        with mock.patch(_HTTP, return_value=_bytes(self.FIXTURE)):
            out = literature.search_crossref("T7 RNA polymerase orthogonal", max_results=5)
        self.assertEqual(out["status"], "success")
        p = out["data"][0]
        self.assertEqual(p["doi"], "10.1021/sb500299c")
        self.assertEqual(p["year"], 2015)
        self.assertEqual(p["crossref_score"], 50.9)
        self.assertNotIn("<", p["abstract"])  # JATS tags stripped
        self.assertIn("T7 variants", p["abstract"])
        self.assertEqual(p["source"], "crossref")


if __name__ == "__main__":
    unittest.main()
