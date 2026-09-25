"""Offline tests for the experimental-measurement extractor (JATS parsing, routing)."""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.tools import measurement_extraction as M
from enzyme_discovery_agent.tool_registry import get_all_tools

# A minimal JATS article: one machine-readable activity table + one image-only table + a fig.
JATS = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <body>
    <table-wrap id="t1">
      <label>Table 1</label>
      <caption><p>Kinetic parameters (kcat, Km) of enzyme variants toward substrate S.</p></caption>
      <table>
        <thead><tr><th>Variant</th><th>kcat (s-1)</th><th>Km (mM)</th><th>Conversion (%)</th></tr></thead>
        <tbody>
          <tr><td>WT</td><td>0.15</td><td>2.3</td><td>48</td></tr>
          <tr><td>R74G</td><td>0.14</td><td>3.4</td><td>91</td></tr>
        </tbody>
      </table>
      <table-wrap-foot><p>Assays at pH 7.5, 37 C.</p></table-wrap-foot>
    </table-wrap>
    <table-wrap id="t2">
      <label>Table 2</label>
      <caption><p>Specific activity screen (image).</p></caption>
      <graphic xlink:href="paper.t002"/>
    </table-wrap>
    <fig id="f1">
      <label>Figure 2</label>
      <caption><p>Bar chart of percent conversion across 25 substrates.</p></caption>
      <graphic xlink:href="paper.f002"/>
    </fig>
  </body>
</article>"""


class TableParsing(unittest.TestCase):
    def test_text_table_structure(self):
        tables = M.parse_jats_tables(JATS)
        self.assertEqual(len(tables), 2)
        t1 = tables[0]
        self.assertEqual(t1.label, "Table 1")
        self.assertFalse(t1.is_image)
        self.assertEqual(t1.header[0], ["Variant", "kcat (s-1)", "Km (mM)", "Conversion (%)"])
        self.assertEqual(t1.rows[0], ["WT", "0.15", "2.3", "48"])
        self.assertEqual(t1.rows[1][0], "R74G")
        self.assertIn("pH 7.5", t1.footnotes)
        self.assertGreater(t1.numeric_density(), 0.5)
        self.assertGreater(t1.relevance(), 1.0)

    def test_image_only_table_flagged(self):
        tables = M.parse_jats_tables(JATS)
        t2 = tables[1]
        self.assertTrue(t2.is_image)
        self.assertEqual(t2.graphic_href, "paper.t002")
        self.assertEqual(t2.rows, [])

    def test_markdown_render(self):
        t1 = M.parse_jats_tables(JATS)[0]
        md = t1.to_markdown()
        self.assertIn("Variant | kcat (s-1)", md)
        self.assertIn("R74G", md)

    def test_figures_parsed(self):
        figs = M.parse_jats_figures(JATS)
        self.assertEqual(len(figs), 1)
        self.assertEqual(figs[0].label, "Figure 2")
        self.assertEqual(figs[0].graphic_href, "paper.f002")
        self.assertGreaterEqual(figs[0].relevance(), 1)  # 'conversion' cue => a data figure

    def test_malformed_xml_is_safe(self):
        self.assertEqual(M.parse_jats_tables(b"<not-xml"), [])
        self.assertEqual(M.parse_jats_figures(b""), [])


class Routing(unittest.TestCase):
    def test_pmcid_resolves_offline(self):
        r = M.resolve_paper("PMC12955871")
        self.assertEqual(r["pmcid"], "PMC12955871")
        self.assertEqual(r["source"], "direct")

    def test_image_asset_urls(self):
        urls = M.image_asset_urls("PMC1847840", "pbio.t002.jpg")
        self.assertIn(M.supp_zip_url("PMC1847840"), urls)          # ZIP is the fetch source
        self.assertTrue(any("pbio.t002.jpg" in u for u in urls))    # member recorded
        self.assertTrue(any("PMC1847840" in u for u in urls))
        self.assertEqual(M.image_asset_urls("", "x"), [])


class FlatPdfRendering(unittest.TestCase):
    """The flat-PDF path renders cue-rich pages for vision instead of truncating to 4k text."""

    def _make_pdf(self):
        import pymupdf
        doc = pymupdf.open()
        p0 = doc.new_page()
        p0.insert_text((72, 72), "Introduction. This enzyme acts on a substrate. No numbers here.")
        p1 = doc.new_page()
        p1.insert_text((72, 72), "Table 2. Steady-state kinetics.\n"
                                 "Enzyme kcat (min-1) Km (mM) kcat/Km (s-1 M-1)\n"
                                 "EnzA 13 20.3 11   EnzB 320 0.005 1.1e6")
        return doc.tobytes()

    def test_selects_the_kinetics_page(self):
        try:
            import pymupdf  # noqa: F401
        except Exception:
            self.skipTest("PyMuPDF not installed")
        imgs = M._pdf_cue_page_images(self._make_pdf())
        self.assertTrue(imgs, "expected at least one cue-bearing page rendered")
        self.assertEqual(imgs[0][0], 1)  # page index 1 (the kinetics table) ranks first
        self.assertTrue(imgs[0][1].startswith("data:image/png;base64,"))

    def test_non_pdf_bytes_return_empty(self):
        self.assertEqual(M._pdf_cue_page_images(b"not a pdf"), [])


class Registration(unittest.TestCase):
    def test_tool_registered_and_generic(self):
        from enzyme_discovery_agent.bootstrap import register_builtin_tools
        register_builtin_tools()
        names = {t["function"]["name"] for t in get_all_tools()}
        self.assertIn("extract_experimental_measurements", names)


if __name__ == "__main__":
    unittest.main()
