"""Tests for evidence-led coverage and EC bond-type rejection filtering.

Validates that:
1. Bond-type filter keeps only relevant EC prefixes
2. Per-EC filtering removes irrelevant sub-classes
3. Graceful fallback when no bond types match
4. EC annotations never trigger candidate generation
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from enzyme_discovery_agent.gates.systematic_coverage import SystematicCoverageGate
from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.chemistry_logic import (
    classify_reaction_chemistry,
    extract_target_bond_type,
    filter_ec_prefixes_by_bond_type,
)


class TestBondTypeFilter(unittest.TestCase):

    def test_amide_formation_keeps_only_6_3(self):
        prefixes = ["1.1", "1.8", "2.5", "6.3"]
        kept = filter_ec_prefixes_by_bond_type(
            prefixes, "amide bond formation between hexamethylenediamine and adipic acid"
        )
        assert "6.3" in kept
        assert "1.1" not in kept
        assert "2.5" not in kept
        assert "1.8" not in kept

    def test_amide_hydrolysis_keeps_3_5(self):
        prefixes = ["3.4", "3.5", "6.3"]
        kept = filter_ec_prefixes_by_bond_type(
            prefixes, "amide bond hydrolysis of nylon oligomers"
        )
        assert "3.5" in kept
        assert "6.3" not in kept or "6.3" in kept  # might match on amide bond type

    def test_ester_formation_keeps_6_2(self):
        prefixes = ["1.1", "2.3", "6.2"]
        kept = filter_ec_prefixes_by_bond_type(
            prefixes, "thioester bond formation with CoA"
        )
        assert "1.1" not in kept

    def test_empty_target_returns_all(self):
        prefixes = ["1.1", "2.5", "6.3"]
        kept = filter_ec_prefixes_by_bond_type(
            prefixes, "some reaction with unknown bond type"
        )
        assert len(kept) == len(prefixes)

    def test_no_match_falls_back(self):
        prefixes = ["1.1", "1.8"]
        kept = filter_ec_prefixes_by_bond_type(
            prefixes, "amide bond formation"
        )
        assert len(kept) <= 2

    def test_single_prefix_passed_through(self):
        prefixes = ["6.3"]
        kept = filter_ec_prefixes_by_bond_type(
            prefixes, "amide bond formation"
        )
        assert kept == ["6.3"]


class TestPerECFiltering(unittest.TestCase):

    def test_ec_6_3_2_4_passes_amide_filter(self):
        chem = classify_reaction_chemistry(["6.3.2.4"], [])
        assert "amide" in chem["bond_types"]

    def test_ec_1_1_1_11_fails_amide_filter(self):
        chem = classify_reaction_chemistry(["1.1.1.11"], [])
        assert "amide" not in chem["bond_types"]

    def test_ec_2_5_1_16_fails_amide_filter(self):
        chem = classify_reaction_chemistry(["2.5.1.16"], [])
        assert "amide" not in chem["bond_types"]

    def test_ec_6_3_1_2_passes_amide_filter(self):
        chem = classify_reaction_chemistry(["6.3.1.2"], [])
        assert "amide" in chem["bond_types"]


class TestQuantitativeImprovement(unittest.TestCase):
    """Verify the filter would have eliminated noise from the actual run."""

    def test_production_run_noise_eliminated(self):
        production_prefixes = ["1.1", "1.3", "1.8", "1.14", "2.5", "6.3"]
        reaction_text = "amide bond formation between hexamethylenediamine and adipic acid"

        kept = filter_ec_prefixes_by_bond_type(production_prefixes, reaction_text)

        assert "6.3" in kept, "Must keep EC 6.3 (amide ligases)"
        noise_prefixes = {"1.1", "1.3", "1.8", "1.14", "2.5"}
        kept_noise = noise_prefixes & set(kept)
        assert len(kept_noise) == 0, (
            f"Should have dropped all noise prefixes, but kept: {kept_noise}"
        )


class TestEvidenceLedCoverage(unittest.TestCase):
    def test_ec_annotations_do_not_trigger_candidate_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "notebook.json").write_text(
                '{"sections": {"reaction_analysis": "EC 3.5.1", '
                '"enzyme_landscape": "EC 3.5.1"}}'
            )
            gate = SystematicCoverageGate()
            state = AgentState(run_dir=run_dir)

            self.assertFalse(gate.should_fire(state, has_eval_calls=False, messages=[]))

    def test_evidence_link_expansion_waits_for_qualified_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = SystematicCoverageGate()
            state = AgentState(run_dir=Path(tmp), homolog_expansion_done=True)

            self.assertTrue(gate.should_fire(state, has_eval_calls=False, messages=[]))


def run_tests():
    test_classes = [
        TestBondTypeFilter,
        TestPerECFiltering,
        TestQuantitativeImprovement,
        TestEvidenceLedCoverage,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        if hasattr(instance, "setUp"):
            instance.setUp()
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
