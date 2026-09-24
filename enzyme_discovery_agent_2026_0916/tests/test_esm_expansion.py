"""Tests for the embedding-driven expansion tool (offline, no GPU/network).

The GPU embedding subprocess, seed-sequence fetch, and UniProt family fetch are
all stubbed, so these exercise the tool's outside-pool retrieval + ESM-rerank +
registration logic deterministically.

Accessions are UniProt-shaped (the tool validates them with a UniProt accession
regex): SEED0x = promising seeds, NEAR0x = family members, INPOOL = a member that
is already in the pool (must never be re-surfaced as a new lead).
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from enzyme_discovery_agent import config
from enzyme_discovery_agent.models import embedding_store as store_mod
from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.tools import esm_expansion as ex

# Deterministic vectors: NEAR01 nearest SEED01; NEAR03 nearest SEED02.
_VECS = {
    "SEED01": [1.0, 0.0, 0.0, 0.0],
    "SEED02": [0.0, 0.0, 1.0, 0.0],
    "NEAR01": [0.98, 0.2, 0.0, 0.0],   # family IPR000111, near SEED01
    "NEAR02": [0.0, 1.0, 0.0, 0.0],    # family IPR000111, far
    "NEAR03": [0.0, 0.0, 0.98, 0.2],   # family IPR000222, near SEED02
    "NEAR04": [0.0, 0.1, 0.9, 0.0],    # family IPR000222, less near SEED02
    "INPOOL": [0.9, 0.1, 0.0, 0.0],    # already in the pool -> must be excluded
}

# What "UniProt" returns as the membership of each family (keyed by the IPR id
# embedded in the query). Includes pool members (SEED01/INPOOL) to prove exclusion.
_FAMILY_MEMBERS = {
    "IPR000111": {"SEED01": "M" + "A" * 30, "INPOOL": "M" + "C" * 30,
                  "NEAR01": "M" + "D" * 30, "NEAR02": "M" + "E" * 30},
    "IPR000222": {"NEAR03": "M" + "F" * 30, "NEAR04": "M" + "G" * 30},
}


def _seed_registry(root: Path):
    update_candidate_registry(root, "SEED01", "evaluated",
                              metadata={"substrate_fit": 0.9, "interpro_family_id": "IPR000111"})
    update_candidate_registry(root, "SEED02", "evaluated",
                              metadata={"substrate_fit": 0.8, "interpro_family_id": "IPR000222"})
    # A pre-existing pool member that is ALSO a family member of IPR000111.
    update_candidate_registry(root, "INPOOL", "discovered", metadata={"source_tool": "search"})


def _fake_embed(run_dir, items, model, max_gpus=8):
    rows = [(it["accession"], it["seq_sha1"], np.array(_VECS[it["accession"]], dtype=np.float32))
            for it in items if it["accession"] in _VECS]
    return rows, "stub embedded"


def _fake_fetch_sequences(run_dir, accessions):
    return {a: "M" + "A" * 40 for a in accessions}


def _fake_uniprot_family_fasta(query, size):
    for ipr, members in _FAMILY_MEMBERS.items():
        if ipr in query:
            return dict(members)
    return {}


class TestFamilyBackground(unittest.TestCase):
    def test_excludes_pool_and_caps(self):
        with mock.patch.object(ex, "_uniprot_family_fasta", _fake_uniprot_family_fasta):
            bg = ex._fetch_family_background("IPR000111", exclude={"SEED01", "INPOOL"}, cap=10)
        self.assertEqual(set(bg), {"NEAR01", "NEAR02"})   # pool members excluded
        self.assertNotIn("SEED01", bg)
        self.assertNotIn("INPOOL", bg)

    def test_cap_limits_count(self):
        with mock.patch.object(ex, "_uniprot_family_fasta", _fake_uniprot_family_fasta):
            bg = ex._fetch_family_background("IPR000111", exclude=set(), cap=1)
        self.assertEqual(len(bg), 1)


class TestExpandCandidatesEsm(unittest.TestCase):
    def test_disabled_returns_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "ENABLE_CANDIDATE_MAP", False):
            out = ex.expand_candidates_esm(run_dir=tmp)
            self.assertEqual(out["status"], "skipped")

    def test_end_to_end_reaches_outside_pool(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "ENABLE_CANDIDATE_MAP", True), \
                mock.patch.object(ex, "embed_sequences", _fake_embed), \
                mock.patch.object(ex, "_fetch_sequences", _fake_fetch_sequences), \
                mock.patch.object(ex, "_uniprot_family_fasta", _fake_uniprot_family_fasta):
            root = Path(tmp)
            _seed_registry(root)
            out = ex.expand_candidates_esm(run_dir=str(root), k=2, n_seeds=2, per_family=50)

            self.assertEqual(out["status"], "success")
            payload = json.loads((root / "candidate_map.json").read_text())
            self.assertEqual(payload["steer_mode"], "expand")
            promising = {p["accession"]: p for p in payload["promising"]}
            self.assertEqual(set(promising), {"SEED01", "SEED02"})

            # ESM decides expansion: SEED01 pulls NEAR01 first, SEED02 pulls NEAR03.
            self.assertEqual(promising["SEED01"]["neighbors"][0]["accession"], "NEAR01")
            self.assertEqual(promising["SEED02"]["neighbors"][0]["accession"], "NEAR03")

            # New leads are genuinely OUTSIDE the prior pool, registered as discovered
            # with expansion provenance + the family they came from.
            reg = load_candidate_registry(root).get("candidates", {})
            self.assertIn("NEAR01", reg)
            self.assertEqual(current_candidate_status(reg["NEAR01"]), "discovered")
            self.assertEqual(reg["NEAR01"]["metadata"]["source_tool"], "esm_expansion:SEED01")
            self.assertEqual(reg["NEAR01"]["metadata"]["expansion_background"], "family:IPR000111")
            self.assertIn("esm_distance", reg["NEAR01"]["metadata"])

            # A pre-existing pool member is never surfaced as an expansion lead.
            all_nbrs = {n["accession"] for p in payload["promising"] for n in p["neighbors"]}
            self.assertNotIn("INPOOL", all_nbrs)
            self.assertNotIn("SEED01", all_nbrs)
            self.assertEqual(reg["INPOOL"]["metadata"]["source_tool"], "search")

            # Selected leads' vectors are persisted so they appear on the map.
            store = store_mod.load_store(root)
            self.assertIn("NEAR01", store.acc_to_row)
            self.assertIn("NEAR03", store.acc_to_row)

    def test_no_gpu_does_not_block_or_register(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "ENABLE_CANDIDATE_MAP", True), \
                mock.patch.object(ex, "embed_sequences",
                                  lambda *a, **k: (None, "no free GPU")), \
                mock.patch.object(ex, "_fetch_sequences", _fake_fetch_sequences), \
                mock.patch.object(ex, "_uniprot_family_fasta", _fake_uniprot_family_fasta):
            root = Path(tmp)
            _seed_registry(root)
            out = ex.expand_candidates_esm(run_dir=str(root), k=2, n_seeds=2)
            self.assertEqual(out["status"], "partial")
            reg = load_candidate_registry(root).get("candidates", {})
            self.assertNotIn("NEAR01", reg)   # nothing registered

    def test_no_families_resolved_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "ENABLE_CANDIDATE_MAP", True), \
                mock.patch.object(ex, "embed_sequences", _fake_embed), \
                mock.patch.object(ex, "_fetch_sequences", _fake_fetch_sequences), \
                mock.patch.object(ex, "_resolve_families", lambda *a, **k: []), \
                mock.patch.object(ex, "_uniprot_family_fasta", _fake_uniprot_family_fasta):
            root = Path(tmp)
            update_candidate_registry(root, "SEED01", "evaluated", metadata={"substrate_fit": 0.9})
            out = ex.expand_candidates_esm(run_dir=str(root), n_seeds=1)
            self.assertEqual(out["status"], "partial")
            self.assertIn("no outside-pool", out["summary"].lower())


if __name__ == "__main__":
    unittest.main()
