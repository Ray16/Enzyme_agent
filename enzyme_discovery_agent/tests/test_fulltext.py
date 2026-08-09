"""Tests for the full-text acquisition layer.

Network is mocked at ``fulltext._get``. The central guarantee under test: a
response is only stored if it VERIFIES as a real PDF (%PDF) or JATS XML -- an
HTML error/landing page is never written as a paper.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_discovery_agent.tools import fulltext as FT

_GET = "enzyme_discovery_agent.tools.fulltext._get"
PDF = b"%PDF-1.5\n%stub pdf bytes\n%%EOF"
HTML = b"<!DOCTYPE html><html><body>Just a moment...</body></html>"
JATS = (b"<article><front></front><body><sec><p>The enzyme uses an "
        b"ATP-grasp mechanism with kcat 5 s-1.</p></sec></body></article>")


class TestVerification(unittest.TestCase):
    def test_pdf_magic(self):
        self.assertTrue(FT._is_pdf(PDF))
        self.assertFalse(FT._is_pdf(HTML))
        self.assertFalse(FT._is_pdf(b""))

    def test_jats_detection(self):
        self.assertTrue(FT._is_jats(JATS))
        self.assertFalse(FT._is_jats(HTML))

    def test_jats_to_text(self):
        txt = FT._jats_to_text(JATS)
        self.assertIn("ATP-grasp mechanism", txt)
        self.assertIn("kcat 5", txt)


class TestFetchOne(unittest.TestCase):
    def test_arxiv_pdf_is_stored(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(_GET, return_value=PDF):
            rec = FT.fetch_one({"title": "T", "arxiv_id": "2401.00001"}, Path(tmp))
            self.assertIsNotNone(rec)
            self.assertEqual(rec["source"], "arxiv")
            self.assertEqual(rec["format"], "pdf")
            self.assertTrue((Path(tmp) / rec["file"]).exists())

    def test_html_masquerade_is_rejected(self):
        # Every source returns an HTML challenge/landing page -> nothing stored.
        with tempfile.TemporaryDirectory() as tmp, mock.patch(_GET, return_value=HTML):
            rec = FT.fetch_one({"title": "T", "arxiv_id": "2401.1", "doi": "10.1/x"},
                               Path(tmp))
            self.assertIsNone(rec)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_europepmc_xml_preferred_and_text_extracted(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(_GET, return_value=JATS):
            rec = FT.fetch_one({"title": "T", "pmcid": "PMC123"}, Path(tmp))
            self.assertEqual(rec["source"], "europepmc")
            self.assertEqual(rec["format"], "xml")
            self.assertGreater(rec["text_chars"], 0)  # XML text extracted for free
            self.assertTrue((Path(tmp) / rec["text_file"]).read_text())

    def test_unpaywall_prefers_repository_pdf(self):
        meta = json.dumps({"oa_locations": [
            {"host_type": "publisher", "url_for_pdf": "https://pub/paywalled.pdf"},
            {"host_type": "repository", "url_for_pdf": "https://repo/green.pdf"},
        ]}).encode()

        def fake_get(url, timeout=60):
            if "api.unpaywall.org" in url:
                return meta
            if url == "https://repo/green.pdf":
                return PDF
            return HTML  # publisher link 403s to HTML in reality

        with tempfile.TemporaryDirectory() as tmp, mock.patch(_GET, side_effect=fake_get):
            rec = FT.fetch_one({"title": "T", "doi": "10.1021/x"}, Path(tmp))
            self.assertIsNotNone(rec)
            self.assertEqual(rec["source"], "unpaywall")

    def test_no_identifier_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(_GET, return_value=PDF):
            self.assertIsNone(FT.fetch_one({"title": "no ids"}, Path(tmp)))


class TestFetchFullTextTool(unittest.TestCase):
    def _make_run(self, tmp, papers):
        lit = Path(tmp) / "artifacts" / "literature"
        lit.mkdir(parents=True)
        (lit / "search_001.json").write_text(json.dumps(
            {"result": {"data": papers}}))

    def test_downloads_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(_GET, return_value=PDF):
            self._make_run(tmp, [{"title": "A", "arxiv_id": "2401.1"}])
            out = FT.fetch_full_text("amide synthetase", run_dir=tmp, max_papers=5)
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["count"], 1)
            manifest = json.loads(
                (Path(tmp) / "artifacts" / "fulltext" / "manifest.json").read_text())
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["source"], "arxiv")

    def test_skips_already_fetched(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(_GET, return_value=PDF):
            self._make_run(tmp, [{"title": "A", "arxiv_id": "2401.1"}])
            FT.fetch_full_text("q", run_dir=tmp, max_papers=5)
            out2 = FT.fetch_full_text("q", run_dir=tmp, max_papers=5)  # nothing new
            self.assertEqual(out2["count"], 0)

    def test_no_literature_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "artifacts" / "literature").mkdir(parents=True)
            out = FT.fetch_full_text("q", run_dir=tmp)
            self.assertEqual(out["status"], "partial")


if __name__ == "__main__":
    unittest.main()
