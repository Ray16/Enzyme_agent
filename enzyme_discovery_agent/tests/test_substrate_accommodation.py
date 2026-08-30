"""Role-aware substrate-accommodation facts.

These pin the behavior the agent-reasoning gates depend on: a target substrate
whose only native counterpart is functional-group-mismatched or cofactor-scale
must be surfaced as UNACCOMMODATED, not averaged away.  The cases use a generic
diacid + free aliphatic diamine target (no benchmark-specific molecules), and
compare three enzyme shapes:

  * a ligase whose amine partner is a FREE aliphatic diamine  -> accommodated
  * a ligase whose amine partner is AMMONIA (cofactor-scale)  -> NOT accommodated
  * a mono-substrate off-target enzyme                        -> NOT accommodated
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.substrate_accommodation import accommodation_facts

# Generic faithful chemistry — a C6 diacid and a C6 free primary diamine.
_DIACID = "O=C(O)CCCCC(=O)O"
_DIAMINE = "NCCCCCCN"
_TARGET = [_DIACID, _DIAMINE]

_AMINO_DIACID = "OC(=O)CCC(N)C(=O)O"   # an amino-diacid (carboxylate donor analogue)
_FREE_DIAMINE_NATIVE = "NCCCCN"        # a shorter free aliphatic diamine
_AMMONIA = "N"                          # one-heavy-atom amine partner
_MONO_ACID = "C(=O)O"                   # a one-carbon acid (mono-substrate enzyme)


def _by_target(facts, smiles):
    return next(p for p in facts["per_target"] if p["target_smiles"] == smiles)


class TestFreeDiamineLigandIsAccommodated(unittest.TestCase):
    def test_diamine_matches_a_free_diamine_native(self):
        facts = accommodation_facts(_TARGET, [_AMINO_DIACID, _FREE_DIAMINE_NATIVE])
        self.assertTrue(facts["all_target_substrates_accommodated"])
        self.assertEqual(facts["unaccommodated_targets"], [])
        amine = _by_target(facts, _DIAMINE)
        self.assertTrue(amine["shares_functional_groups"])
        self.assertFalse(amine["native_is_cofactor_scale"])


class TestAmmoniaPartnerIsNotAccommodated(unittest.TestCase):
    def test_diamine_forced_onto_ammonia_is_flagged(self):
        # One-to-one assignment forces the diamine onto ammonia (the acid takes
        # the amino-diacid), exposing that the diamine has no genuine partner.
        facts = accommodation_facts(_TARGET, [_AMINO_DIACID, _AMMONIA])
        self.assertIn(_DIAMINE, facts["unaccommodated_targets"])
        self.assertFalse(facts["all_target_substrates_accommodated"])
        amine = _by_target(facts, _DIAMINE)
        self.assertEqual(amine["native_smiles"], _AMMONIA)
        self.assertTrue(amine["native_is_cofactor_scale"])
        self.assertEqual(amine["native_heavy_atoms"], 1)
        self.assertFalse(amine["shares_functional_groups"])
        # The acid partner is still genuinely matched — the failure is specific.
        acid = _by_target(facts, _DIACID)
        self.assertTrue(acid["shares_functional_groups"])

    def test_ammonia_case_is_not_rescued_by_size_similarity(self):
        # The whole point: even though the amino-diacid is a good size/shape match
        # for the diacid, the diamine remains unaccommodated — no blended mean
        # hides it.
        facts = accommodation_facts(_TARGET, [_AMINO_DIACID, _AMMONIA])
        self.assertFalse(facts["all_target_substrates_accommodated"])


class TestMonoSubstrateEnzymeCannotHostBoth(unittest.TestCase):
    def test_missing_counterpart_is_unaccommodated(self):
        facts = accommodation_facts(_TARGET, [_MONO_ACID])
        # Only one native, two target substrates -> one target gets no counterpart.
        self.assertFalse(facts["all_target_substrates_accommodated"])
        self.assertEqual(len(facts["unaccommodated_targets"]), 1)
        missing = next(p for p in facts["per_target"] if p["native_smiles"] is None)
        self.assertFalse(missing["shares_functional_groups"])


class TestDegenerateInputs(unittest.TestCase):
    def test_no_targets_returns_none(self):
        self.assertIsNone(accommodation_facts([], [_AMMONIA]))

    def test_no_natives_leaves_all_unaccommodated(self):
        facts = accommodation_facts(_TARGET, [])
        self.assertEqual(len(facts["unaccommodated_targets"]), 2)
        self.assertFalse(facts["all_target_substrates_accommodated"])


if __name__ == "__main__":
    unittest.main()
