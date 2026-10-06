"""Offline tests for the mechanism extractor's paper ranking + confidence gating.

These cover the pure decision logic that fixes the "first hit is a physiology paper that
mentions the enzyme, so a guessed (water) nucleophile is returned with low confidence and
still accepted" failure. No network / LLM calls.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_discovery_agent.tools import mechanism_extraction as MECH
from enzyme_discovery_agent.tool_registry import get_all_tools
from enzyme_discovery_agent.models.candidate_registry import (
    load_candidate_registry,
    update_candidate_registry,
)


class PaperRanking(unittest.TestCase):
    def test_mechanism_paper_outranks_physiology_paper(self):
        papers = [
            {"title": "HadS, a 2-haloacid dehalogenase enhances saline-alkaline tolerance "
                      "when heterologously expressed in E. coli", "is_oa": True, "pmcid": "PMC1"},
            {"title": "Crystal structure and catalytic mechanism of L-2-haloacid dehalogenase: "
                      "the aspartate nucleophile and active site", "is_oa": True, "pmcid": "PMC2"},
        ]
        ranked = MECH._rank_mechanism_papers(papers)
        self.assertEqual(ranked[0]["pmcid"], "PMC2")  # the mechanism/structure paper wins

    def test_open_access_preferred_over_closed(self):
        papers = [
            {"title": "Reaction mechanism of the enzyme", "is_oa": False, "pmcid": ""},
            {"title": "Reaction mechanism of the enzyme", "is_oa": True, "pmcid": "PMC9"},
        ]
        ranked = MECH._rank_mechanism_papers(papers)
        self.assertTrue(ranked[0]["is_oa"])  # can only extract from fetchable full text


class FactScoring(unittest.TestCase):
    def _asp(self):
        return {"confidence": "medium",
                "nucleophile": {"source": "protein_residue", "identity": "Asp carboxylate oxygen"},
                "catalytic_residues": [{"residue": "Asp", "role": "nucleophile"}],
                "reference_structures": ["1JUD"],
                "evidence": "The aspartate attacks C2 in an SN2 displacement."}

    def _water_lowconf(self):
        return {"confidence": "low",
                "nucleophile": {"source": "water", "identity": "water molecule"},
                "catalytic_residues": [],
                "reference_structures": [],
                "evidence": "No specific catalytic residues or mechanistic details are provided in this text."}

    def test_grounded_asp_beats_water_fallback(self):
        s_good, g_good = MECH._score_facts(self._asp())
        s_bad, g_bad = MECH._score_facts(self._water_lowconf())
        self.assertTrue(g_good)
        self.assertFalse(g_bad)
        self.assertGreater(s_good, s_bad)

    def test_disclaimer_evidence_is_penalized(self):
        base = self._asp()
        base_score, _ = MECH._score_facts(base)
        disclaimed = dict(base, evidence="No mechanistic detail is provided in this text.")
        dscore, _ = MECH._score_facts(disclaimed)
        self.assertLess(dscore, base_score)

    def test_empty_or_nonjson_is_not_grounded(self):
        self.assertFalse(MECH._score_facts({})[1])
        self.assertFalse(MECH._score_facts(None)[1])

    def test_redox_with_null_nucleophile_is_grounded(self):
        """A non-substitution enzyme (hydride transfer) has NO nucleophile — it must still
        score grounded on its general facts, not get penalized for the null."""
        redox = {"confidence": "medium",
                 "mechanism_class": "redox_hydride",
                 "nucleophile": {"source": None, "identity": None},
                 "participants": [
                     {"role": "hydride_donor", "identity": "substrate C1", "on": "substrate"},
                     {"role": "hydride_acceptor", "identity": "NAD+ C4", "on": "cofactor"}],
                 "bonds_broken": [{"atoms": "C1-H", "on": "substrate"}],
                 "bonds_formed": [{"atoms": "C4-H(NAD)", "on": "cofactor"}],
                 "catalytic_residues": [{"residue": "His51", "role": "base"}],
                 "cofactors": [{"name": "NAD+", "count": 1}],
                 "metals": [{"name": "Zn", "count": 1}],
                 "reference_structures": ["1ADB"],
                 "evidence": "Hydride is transferred from the substrate to NAD+."}
        score, grounded = MECH._score_facts(redox)
        self.assertTrue(grounded, f"redox facts should be grounded (score={score})")


class FederatedDiscovery(unittest.TestCase):
    """Candidate discovery goes through the agent's multi-source search + consolidate_pool,
    not a hand-rolled single-source query."""

    def test_to_ref_needs_pmcid_or_doi(self):
        self.assertIsNone(MECH._to_ref({"title": "no ids here"}))
        self.assertEqual(MECH._to_ref({"doi": "10.1/z"})["doi"], "10.1/z")
        self.assertEqual(MECH._to_ref({"pmcid": "pmc9"})["pmcid"], "PMC9")

    def test_discover_uses_all_sources_and_backfilled_pmcid(self):
        from unittest import mock
        from enzyme_discovery_agent.tools import literature as lit

        called = set()

        def src(name):
            def _f(query, max_results=10, rerank_query=""):
                called.add(name)
                return {"status": "ok",
                        "data": [{"title": "Crystal structure and mechanism of X",
                                  "doi": "10.1/x", "pmcid": "", "sources": [name]}]}
            return _f

        def consolidate(query, papers, max_results=25, min_score=0.25):
            # emulate the cross-source merge back-filling a PMCID onto the DOI-only hit
            return {"ranked": [{"title": "Crystal structure and mechanism of X",
                                "doi": "10.1/x", "pmcid": "PMC7",
                                "sources": ["semantic_scholar", "europepmc"]}],
                    "funnel": {}}

        with mock.patch.multiple(lit,
                                 search_pubmed=src("pubmed"),
                                 search_openalex=src("openalex"),
                                 search_europepmc=src("europepmc"),
                                 search_semantic_scholar=src("semantic_scholar"),
                                 consolidate_pool=consolidate):
            refs = MECH._discover_candidates("enzyme X")

        self.assertEqual(called, {"pubmed", "openalex", "europepmc", "semantic_scholar"})
        self.assertTrue(refs)
        self.assertEqual(refs[0]["pmcid"], "PMC7")  # back-filled → fetchable full text


class Registration(unittest.TestCase):
    def test_tool_registered(self):
        from enzyme_discovery_agent.bootstrap import register_builtin_tools
        register_builtin_tools()
        names = {t["function"]["name"] for t in get_all_tools()}
        self.assertIn("extract_reaction_mechanism", names)


class RegistryIntegration(unittest.TestCase):
    def test_grounded_result_is_attached_to_exact_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "evaluated", metadata={
                "gene_names": ["ExampleA"],
            })
            result = {
                "status": "ok", "schema_version": "1.0",
                "data": {"mechanism_class": "hydrolysis", "confidence": "high"},
                "paper": {"doi": "10.1/example"}, "papers_tried": [],
            }
            MECH._persist_mechanism_evidence(run_dir, "P12345", result)
            metadata = load_candidate_registry(run_dir)["candidates"]["P12345"]["metadata"]
            self.assertEqual(metadata["mechanism_evidence"]["status"], "ok")
            self.assertEqual(metadata["mechanism_evidence"]["facts"]["mechanism_class"],
                             "hydrolysis")

    def test_ambiguous_name_is_not_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for accession in ("P12345", "Q12345"):
                update_candidate_registry(run_dir, accession, "evaluated", metadata={
                    "gene_names": ["same"],
                })
            MECH._persist_mechanism_evidence(run_dir, "same", {
                "status": "ok", "data": {"mechanism_class": "other"},
            })
            registry = load_candidate_registry(run_dir)["candidates"]
            self.assertNotIn("mechanism_evidence", registry["P12345"]["metadata"])
            self.assertNotIn("mechanism_evidence", registry["Q12345"]["metadata"])

    def test_cache_key_uses_only_resolved_catalytic_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "evaluated", metadata={
                "interpro_family_id": "IPR_CATALYTIC",
                "domain_architecture": [
                    {"accession": "IPR_ACCESSORY", "type": "domain"},
                ],
            })
            self.assertEqual(
                MECH._candidate_family_ids(run_dir, "P12345"),
                ["IPR_CATALYTIC"],
            )


class CacheAdmission(unittest.TestCase):
    def _facts(self, *, confidence="high"):
        return {
            "mechanism_class": "other",
            "confidence": confidence,
            "participants": [{"role": "other", "identity": "X"}],
            "bonds_formed": [{"atoms": "A-B"}],
            "catalytic_residues": [{"residue": "A1"}],
            "reference_structures": [],
            "nucleophile": None,
            "electrophile": None,
            "leaving_group": None,
            "steps": [{"index": 1, "name": "step", "evidence": "quoted"}],
            "evidence": "A supported chemical step.",
        }

    def _run(self, run_dir, facts):
        update_candidate_registry(
            run_dir, "P12345", "evaluated",
            metadata={"interpro_family_id": "IPR000001"},
        )
        paper = {"pmcid": "PMC1", "doi": "10.1/example", "title": "Mechanism"}
        with mock.patch.object(MECH, "resolve_paper", return_value=paper), \
             mock.patch.object(MECH, "_extract_from", return_value=facts):
            return MECH.extract_reaction_mechanism(
                "P12345", paper="PMC1", llm_client=object(), run_dir=run_dir,
            )

    def test_grounded_schema_valid_steps_are_cached(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch(
                 "enzyme_discovery_agent.tools.mechanism_cache.MechanismCache.put"
             ) as put:
            self._run(Path(tmp), self._facts())
            put.assert_called_once()
            self.assertEqual(put.call_args.args[0], ["IPR000001"])

    def test_low_confidence_steps_are_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("enzyme_discovery_agent.tools.mechanism_cache.MechanismCache.put") as put:
            self._run(Path(tmp), self._facts(confidence="low"))
            put.assert_not_called()

    def test_schema_invalid_steps_are_not_cached(self):
        facts = self._facts()
        facts["steps"] = "not-a-list"
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("enzyme_discovery_agent.tools.mechanism_cache.MechanismCache.put") as put:
            self._run(Path(tmp), facts)
            put.assert_not_called()


if __name__ == "__main__":
    unittest.main()
