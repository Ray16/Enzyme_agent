"""Saturation library construction, and the guards that keep ids trustworthy."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from enzyme_design_agent.saturation.variants import (
    AA20, _check_unique, build_variants, job_counts, mutate,
    write_variants_fasta, write_variants_tsv,
)

SEQ = "ASGVT"


class MutateTest(unittest.TestCase):
    def test_applies_by_one_based_resnum(self):
        self.assertEqual("ACGVT", mutate(SEQ, 2, "S", "C"))

    def test_wild_type_mismatch_is_refused(self):
        """The check that catches a renumbered structure before it ruins a campaign."""
        with self.assertRaisesRegex(ValueError, "structure and the sequence disagree"):
            mutate(SEQ, 2, "A", "C")

    def test_out_of_range_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside a sequence"):
            mutate(SEQ, 99, "A", "C")

    def test_non_substitution_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a substitution"):
            mutate(SEQ, 2, "S", "S")


class BuildVariantsTest(unittest.TestCase):
    def test_nineteen_per_position_plus_wt(self):
        vs = build_variants(SEQ, [(2, "S"), (3, "G")])
        self.assertEqual(1 + 2 * 19, len(vs))
        self.assertEqual(1, sum(1 for v in vs if v.kind == "wt"))

    def test_wild_type_is_never_among_the_substitutions(self):
        vs = build_variants(SEQ, [(2, "S")])
        self.assertNotIn("S", [v.mut_aa for v in vs if v.kind == "single"])

    def test_ids_follow_the_merge_contract(self):
        vs = build_variants(SEQ, [(2, "S")], label="enz")
        ids = {v.id for v in vs}
        self.assertIn("enz_WT", ids)
        self.assertIn("enz_S2C", ids)

    def test_label_is_optional(self):
        vs = build_variants(SEQ, [(2, "S")])
        self.assertIn("S2C", {v.id for v in vs})

    def test_sequences_actually_carry_the_mutation(self):
        vs = build_variants(SEQ, [(2, "S")])
        v = next(v for v in vs if v.mutation == "S2C")
        self.assertEqual("ACGVT", v.seq)
        self.assertEqual(1, sum(a != b for a, b in zip(SEQ, v.seq)))

    def test_restricted_alphabet(self):
        vs = build_variants(SEQ, [(2, "S")], alphabet="AC", include_wt=False)
        self.assertEqual({"A", "C"}, {v.mut_aa for v in vs})

    def test_accepts_pocket_positions(self):
        class P:
            def __init__(self, rn, aa):
                self.resnum, self.wt_aa = rn, aa
        vs = build_variants(SEQ, [P(2, "S")], include_wt=False)
        self.assertEqual(19, len(vs))

    def test_annotation_records_but_never_excludes(self):
        """Catalytic positions stay in the library; that is the whole design."""
        vs = build_variants(SEQ, [(2, "S"), (3, "G")],
                            annotate=lambda rn, wt: "catalytic" if rn == 2 else "")
        at2 = [v for v in vs if v.resnum == 2]
        self.assertEqual(19, len(at2))
        self.assertTrue(all(v.annotation == "catalytic" for v in at2))

    def test_repeating_a_position_is_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate variant ids"):
            build_variants(SEQ, [(2, "S"), (2, "S")])

    def test_duplicate_sequences_are_refused(self):
        """Defensive: two ids folding to one sequence would average two labels.

        ``build_variants`` cannot currently produce this -- a repeated position
        trips the id check first -- so the guard is asserted at its own level
        rather than through a contrived call that no longer reaches it.
        """
        v = build_variants(SEQ, [(2, "S")], include_wt=False)[0]
        with self.assertRaisesRegex(ValueError, "identical sequences"):
            _check_unique([v, replace(v, id=v.id + "_copy")])

    def test_modes_default_to_needed(self):
        vs = build_variants(SEQ, [(2, "S")], modes=("apo", "holo"))
        self.assertEqual(("apo", "holo"), vs[0].needed_modes)

    def test_dedup_marks_existing_pairs_not_needed(self):
        vs = build_variants(SEQ, [(2, "S")], label="e", modes=("apo", "holo"),
                            existing={("e_S2C", "apo")}, dedup=True)
        v = next(v for v in vs if v.id == "e_S2C")
        self.assertEqual(("holo",), v.needed_modes)

    def test_dedup_off_by_default_rebuilds_everything(self):
        """Uniformity over thrift: one campaign, one protocol."""
        vs = build_variants(SEQ, [(2, "S")], label="e", modes=("apo",),
                            existing={("e_S2C", "apo")})
        v = next(v for v in vs if v.id == "e_S2C")
        self.assertEqual(("apo",), v.needed_modes)


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.vs = build_variants(SEQ, [(2, "S")], label="e", modes=("apo", "holo"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_tsv_has_a_need_column_per_mode(self):
        p = write_variants_tsv(self.vs, self.d / "v.tsv", modes=("apo", "holo"))
        rows = list(csv.DictReader(p.open(), delimiter="\t"))
        self.assertIn("need_apo", rows[0])
        self.assertIn("need_holo", rows[0])
        self.assertEqual(len(self.vs), len(rows))

    def test_fasta_has_one_record_per_variant_not_per_mode(self):
        p = write_variants_fasta(self.vs, self.d / "v.fasta")
        self.assertEqual(len(self.vs), p.read_text().count(">"))

    def test_fasta_skips_variants_needing_nothing(self):
        vs = build_variants(SEQ, [(2, "S")], label="e", modes=("apo",),
                            existing={("e_S2C", "apo")}, dedup=True)
        p = write_variants_fasta(vs, self.d / "v2.fasta")
        self.assertNotIn(">e_S2C", p.read_text())

    def test_job_counts_multiply_by_samples(self):
        c = job_counts(self.vs, ("apo", "holo"), n_samples=50)
        self.assertEqual(2 * len(self.vs), c["n_jobs"])
        self.assertEqual(50 * 2 * len(self.vs), c["n_models"])
        self.assertEqual(1, c["n_positions"])

    def test_alphabet_default_is_all_twenty(self):
        self.assertEqual(20, len(AA20))
        self.assertEqual(20, len(set(AA20)))


if __name__ == "__main__":
    unittest.main()
