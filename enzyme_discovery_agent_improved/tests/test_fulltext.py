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

from enzyme_discovery_agent_improved.tools import fulltext as FT

_GET = "enzyme_discovery_agent_improved.tools.fulltext._get"
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


class TestHarvestCap(unittest.TestCase):
    """The initial full-text harvest reads the top-N most-relevant papers, not the
    whole (~10^2) pool — enough to orient, bounded so the early stage searches for
    candidates rather than reading exhaustively."""

    def test_caps_to_top_n_in_rank_order(self):
        ranked = [{"doi": f"10.1/{i}"} for i in range(100)]
        sel = FT._select_harvest_papers(ranked, set(), cap=40)
        self.assertEqual(len(sel), 40)
        self.assertEqual(sel[0]["doi"], "10.1/0")   # most-relevant first
        self.assertEqual(sel[-1]["doi"], "10.1/39")

    def test_skips_already_fetched(self):
        ranked = [{"doi": f"10.1/{i}"} for i in range(10)]
        sel = FT._select_harvest_papers(ranked, {"10.1/0", "10.1/1"}, cap=40)
        self.assertNotIn("10.1/0", [p["doi"] for p in sel])
        self.assertEqual(len(sel), 8)

    def test_skips_papers_without_a_key(self):
        ranked = [{"title": "no id"}, {"doi": "10.1/9"}]
        sel = FT._select_harvest_papers(ranked, set(), cap=40)
        self.assertEqual([p.get("doi") for p in sel], ["10.1/9"])

    def test_cap_below_pool_size_bounds_reads(self):
        ranked = [{"pmid": str(i)} for i in range(200)]
        self.assertEqual(len(FT._select_harvest_papers(ranked, set(), cap=40)), 40)


class TestLeanHarvest(unittest.TestCase):
    """The rescoped harvest reads only a few DOWNLOADABLE papers (paywalls skipped).
    Its primary, always-on channel is regex DEPOSITED-accession mining of the
    primary papers; the LLM named/cited extraction is OFF by default."""

    def _run(self, tmp, papers, fetch_map, target, llm_named=True):
        lit = Path(tmp) / "artifacts" / "literature"
        lit.mkdir(parents=True)
        (lit / "search_001.json").write_text(json.dumps({"result": {"data": papers}}))
        from enzyme_discovery_agent_improved import config
        named_seen = {}

        def fake_fetch_one(paper, dest, basename=None):
            key = FT._fulltext_key(paper)
            if not fetch_map.get(key):          # paywalled -> None (fast-fail)
                return None
            return {"key": key, "title": paper.get("title", ""), "source": "europepmc",
                    "format": "xml", "text_file": None}

        def fake_named(run, query, dest, manifest, llm, register_fn, already, remaining):
            named_seen["n"] = len(manifest)    # how many papers reached extraction
            return []

        with mock.patch.object(config, "FULLTEXT_HARVEST_MAX_PAPERS", target), \
             mock.patch.object(config, "FULLTEXT_LLM_NAMED_EXTRACTION", llm_named), \
             mock.patch.object(FT, "consolidate_pool",
                               lambda q, ps, max_results=25: {"ranked": ps}), \
             mock.patch.object(FT, "fetch_one", fake_fetch_one), \
             mock.patch.object(FT, "_register_named_enzymes_from_fulltext", fake_named), \
             mock.patch("enzyme_discovery_agent_improved.models.candidate_registry."
                        "update_candidate_registry") as reg:
            out = FT.mine_and_register_fulltext_accessions(
                tmp, "q", llm_client=object())
        return out, named_seen, reg

    def test_reads_until_target_and_skips_paywalls(self):
        # 2 paywalled papers rank ABOVE 3 open-access ones; target=2 OA reads.
        # (LLM named extraction enabled here only to count papers reached.)
        papers = [{"doi": f"10.1/{i}", "title": f"P{i}"} for i in range(5)]
        fetch_map = {"10.1/0": False, "10.1/1": False,   # paywalled
                     "10.1/2": True, "10.1/3": True, "10.1/4": True}  # OA
        with tempfile.TemporaryDirectory() as tmp:
            _, named_seen, _ = self._run(tmp, papers, fetch_map, target=2, llm_named=True)
            # reached extraction with exactly the 2 OA papers (paywalls skipped, not
            # counted against the target)
            self.assertEqual(named_seen["n"], 2)

    def test_llm_named_extraction_off_by_default(self):
        # With the default flag OFF, the expensive LLM named/cited extraction is
        # not invoked at all.
        papers = [{"doi": "10.1/2", "title": "P2"}]
        with tempfile.TemporaryDirectory() as tmp:
            _, named_seen, _ = self._run(tmp, papers, {"10.1/2": True},
                                         target=6, llm_named=False)
            self.assertNotIn("n", named_seen)  # never called

    def test_deposit_accession_mining_registers_from_primary_paper(self):
        # A downloaded PRIMARY paper whose text carries a deposited accession gets
        # that accession resolved (deterministically) and registered — the always-on,
        # LLM-free literature channel.
        from enzyme_discovery_agent_improved import config
        lit = Path  # noqa
        papers = [{"doi": "10.1/7", "title": "A primary amide synthetase study"}]
        with tempfile.TemporaryDirectory() as tmp:
            litdir = Path(tmp) / "artifacts" / "literature"
            litdir.mkdir(parents=True)
            (litdir / "s.json").write_text(json.dumps({"result": {"data": papers}}))

            def fake_fetch_one(paper, dest, basename=None):
                # write a text file containing a deposited accession
                (dest / "txt").mkdir(parents=True, exist_ok=True)
                tf = "txt/p7.txt"
                (dest / tf).write_text("Sequence deposited as WP_073481158.1 in this work.")
                return {"key": "10.1/7", "title": paper["title"], "source": "epmc",
                        "format": "xml", "text_file": tf}

            with mock.patch.object(config, "FULLTEXT_LLM_NAMED_EXTRACTION", False), \
                 mock.patch.object(FT, "consolidate_pool",
                                   lambda q, ps, max_results=25: {"ranked": ps}), \
                 mock.patch.object(FT, "fetch_one", fake_fetch_one), \
                 mock.patch.object(FT, "_resolve_mined_accession",
                                   lambda a: ["A0A1M5ABR5"] if a.startswith("WP_073481158") else []), \
                 mock.patch("enzyme_discovery_agent_improved.models.candidate_registry."
                            "update_candidate_registry") as reg:
                out = FT.mine_and_register_fulltext_accessions(tmp, "q", llm_client=object())

            self.assertIn("A0A1M5ABR5", out)
            sources = [c.kwargs.get("metadata", {}).get("source_tool", "")
                       for c in reg.call_args_list]
            self.assertIn("fulltext_deposited_accession", sources)

    def test_reviews_are_skipped_by_deposit_mining(self):
        # A review paper is NOT deposit-mined (reviews cite many off-target seqs).
        from enzyme_discovery_agent_improved import config
        papers = [{"doi": "10.1/8", "title": "Amide bond formation: a review"}]
        with tempfile.TemporaryDirectory() as tmp:
            litdir = Path(tmp) / "artifacts" / "literature"
            litdir.mkdir(parents=True)
            (litdir / "s.json").write_text(json.dumps({"result": {"data": papers}}))

            def fake_fetch_one(paper, dest, basename=None):
                (dest / "txt").mkdir(parents=True, exist_ok=True)
                (dest / "txt/p8.txt").write_text("cites WP_073481158.1 as an example.")
                return {"key": "10.1/8", "title": paper["title"], "source": "epmc",
                        "format": "xml", "text_file": "txt/p8.txt"}

            resolve_calls = []
            with mock.patch.object(config, "FULLTEXT_LLM_NAMED_EXTRACTION", False), \
                 mock.patch.object(FT, "consolidate_pool",
                                   lambda q, ps, max_results=25: {"ranked": ps}), \
                 mock.patch.object(FT, "fetch_one", fake_fetch_one), \
                 mock.patch.object(FT, "_resolve_mined_accession",
                                   lambda a: resolve_calls.append(a) or []), \
                 mock.patch("enzyme_discovery_agent_improved.models.candidate_registry."
                            "update_candidate_registry"):
                FT.mine_and_register_fulltext_accessions(tmp, "q", llm_client=object())
            self.assertEqual(resolve_calls, [])  # review skipped -> nothing mined
