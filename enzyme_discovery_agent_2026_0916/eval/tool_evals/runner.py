"""Runner for the per-tool eval harness. See package docstring for usage."""

from __future__ import annotations

import argparse
import re
import sys
import time

from enzyme_discovery_agent.tools.mechanism_extraction import extract_reaction_mechanism
from enzyme_discovery_agent.tools.measurement_extraction import extract_experimental_measurements
from enzyme_discovery_agent.tools.mechanism_schema import validate_mechanism_facts
from enzyme_discovery_agent.eval.tool_evals.cases import CASES

_CONF_ORDER = {"low": 0, "medium": 1, "high": 2}


def _fn(tool):
    """Unwrap a @tool-decorated function to its plain callable."""
    return getattr(tool, "__wrapped__", tool)


def _check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    return (name, bool(ok), detail)


def _eval_mechanism(case: dict) -> tuple[list[tuple[str, bool, str]], dict]:
    result = _fn(extract_reaction_mechanism)(case["enzyme"], paper=case.get("paper", ""))
    data = result.get("data") or {}
    exp = case["expect"]
    checks = [
        _check("fetched+extracted", bool(data), f"status={result.get('status')}"),
        _check("schema_valid", not validate_mechanism_facts(data) if data else False,
               str(validate_mechanism_facts(data)[:2]) if data else "no data"),
    ]
    if data:
        if "mechanism_class_in" in exp:
            v = data.get("mechanism_class")
            checks.append(_check("mechanism_class", v in exp["mechanism_class_in"],
                                 f"{v!r} in {exp['mechanism_class_in']}"))
        if "is_clean_inline_substitution" in exp:
            v = data.get("is_clean_inline_substitution")
            checks.append(_check("is_clean_inline", v == exp["is_clean_inline_substitution"],
                                 f"{v!r} == {exp['is_clean_inline_substitution']!r}"))
        if "participant_roles_any" in exp:
            roles = {(p or {}).get("role") for p in (data.get("participants") or [])}
            want = set(exp["participant_roles_any"])
            checks.append(_check("participant_roles", bool(roles & want),
                                 f"{sorted(r for r in roles if r)} ∩ {sorted(want)}"))
        if "nucleophile_source_in" in exp:
            src = (data.get("nucleophile") or {}).get("source")
            checks.append(_check("nucleophile_source", src in exp["nucleophile_source_in"],
                                 f"{src!r}"))
        if exp.get("nucleophile_null"):
            nu = data.get("nucleophile") or {}
            is_null = not (nu.get("identity") or "").strip()
            checks.append(_check("nucleophile_null", is_null,
                                 f"identity={nu.get('identity')!r}"))
        if exp.get("reactive_pair_present"):
            rp = data.get("reactive_pair") or {}
            checks.append(_check("reactive_pair", bool(rp.get("atom_a") and rp.get("atom_b")),
                                 f"a={rp.get('atom_a')!r} b={rp.get('atom_b')!r}"))
        if "catalytic_residues_regex" in exp:
            blob = " ".join(str((r or {}).get("residue", "")) for r in
                            (data.get("catalytic_residues") or []))
            hit = bool(re.search(exp["catalytic_residues_regex"], blob))
            checks.append(_check("catalytic_residues", hit, blob[:50]))
        if "metals_regex" in exp:
            blob = " ".join(str((m or {}).get("name", "")) for m in (data.get("metals") or []))
            checks.append(_check("metals", bool(re.search(exp["metals_regex"], blob)), blob[:40]))
        if "min_confidence" in exp:
            conf = (data.get("confidence") or "low")
            checks.append(_check("min_confidence",
                                 _CONF_ORDER.get(conf, -1) >= _CONF_ORDER[exp["min_confidence"]],
                                 f"{conf} >= {exp['min_confidence']}"))
        # --- modeling-field correctness checks ---
        if "stereochemistry_in" in exp:
            v = data.get("stereochemistry")
            checks.append(_check("stereochemistry", v in exp["stereochemistry_in"],
                                 f"{v!r} in {exp['stereochemistry_in']}"))
        if "covalent_intermediate_forms" in exp:
            v = (data.get("covalent_intermediate") or {}).get("forms")
            checks.append(_check("covalent_forms", v == exp["covalent_intermediate_forms"],
                                 f"{v!r} == {exp['covalent_intermediate_forms']!r}"))
        if "covalent_residue_regex" in exp:
            v = str((data.get("covalent_intermediate") or {}).get("residue") or "")
            checks.append(_check("covalent_residue",
                                 bool(re.search(exp["covalent_residue_regex"], v, re.I)), v[:30]))
        if "substrate_reactive_atom_regex" in exp:
            v = str((data.get("substrate") or {}).get("reactive_atom") or "")
            checks.append(_check("substrate_reactive_atom",
                                 bool(re.search(exp["substrate_reactive_atom_regex"], v, re.I)),
                                 v[:38]))
        if exp.get("michaelis_structure_present"):
            v = data.get("michaelis_structure")
            checks.append(_check("michaelis_structure", bool(v), f"{v!r}"))
        if exp.get("reference_structures_present"):
            v = data.get("reference_structures") or []
            checks.append(_check("reference_structures", bool(v), f"{v!r}"))
        if "cofactor_ccd_includes" in exp:
            ccds = {str((c or {}).get("ccd", "")).upper() for c in (data.get("cofactors") or [])}
            want = {x.upper() for x in exp["cofactor_ccd_includes"]}
            checks.append(_check("cofactor_ccd", bool(ccds & want),
                                 f"{sorted(c for c in ccds if c)} ⊇ {sorted(want)}"))
        if "metal_ccd_includes" in exp:
            ccds = {str((m or {}).get("ccd", "")).upper() for m in (data.get("metals") or [])}
            want = {x.upper() for x in exp["metal_ccd_includes"]}
            checks.append(_check("metal_ccd", bool(ccds & want),
                                 f"{sorted(c for c in ccds if c)} ⊇ {sorted(want)}"))
    info = {"status": result.get("status"), "confidence": data.get("confidence"),
            "paper": (result.get("paper") or {}).get("title", "")[:52]}
    return checks, info


def _eval_measurement(case: dict) -> tuple[list[tuple[str, bool, str]], dict]:
    result = _fn(extract_experimental_measurements)(
        case["identifier"], context_hint=case.get("context_hint", ""))
    rows = result.get("data") or []
    exp = case["expect"]
    checks = [_check("extracted_rows", bool(rows), f"count={len(rows)} status={result.get('status')}")]
    if "min_count" in exp:
        checks.append(_check("min_count", len(rows) >= exp["min_count"],
                             f"{len(rows)} >= {exp['min_count']}"))
    for spec in exp.get("rows_include", []):
        found = None
        for m in rows:
            if not re.search(spec["enzyme_re"], str(m.get("enzyme", "")), re.I):
                continue
            if not re.search(spec["substrate_re"], str(m.get("substrate", "")), re.I):
                continue
            if str(m.get("metric", "")).lower() != spec["metric"].lower():
                continue
            try:
                if abs(float(m.get("value")) - spec["value"]) <= spec["tol"]:
                    found = m
                    break
            except (TypeError, ValueError):
                continue
        label = f"{spec['enzyme_re']}/{spec['substrate_re']}/{spec['metric']}≈{spec['value']}"
        checks.append(_check(f"row[{label}]", found is not None,
                             f"got {found.get('value') if found else 'no match'}"))
    info = {"status": result.get("status"), "count": len(rows),
            "paper": (result.get("paper") or {}).get("pmcid", "")}
    return checks, info


def run_case(case: dict) -> tuple[bool, list[tuple[str, bool, str]], dict, float]:
    t0 = time.time()
    if case["kind"] == "mechanism":
        checks, info = _eval_mechanism(case)
    else:
        checks, info = _eval_measurement(case)
    dt = time.time() - t0
    passed = all(ok for _n, ok, _d in checks)
    return passed, checks, info, dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-tool eval harness (isolated tool testing).")
    ap.add_argument("--case", help="substring filter on case name", default="")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--schema-selfcheck", action="store_true",
                    help="offline: validate the schema validator on fixtures, no network")
    args = ap.parse_args(argv)

    if args.list:
        for c in CASES:
            print(f"  [{c['kind']:11}] {c['name']}")
        return 0

    if args.schema_selfcheck:
        good = {"mechanism_class": "SN2", "confidence": "high",
                "participants": [{"role": "nucleophile", "identity": "Asp"}]}
        bad = {"mechanism_class": "not_a_class", "confidence": "sure",
               "participants": "should-be-a-list"}
        gv, bv = validate_mechanism_facts(good), validate_mechanism_facts(bad)
        print(f"valid fixture  -> {len(gv)} violations {gv}")
        print(f"invalid fixture-> {len(bv)} violations {bv}")
        ok = (not gv) and len(bv) >= 2
        print("SCHEMA SELFCHECK:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    cases = [c for c in CASES if args.case.lower() in c["name"].lower()]
    if not cases:
        print(f"no cases match {args.case!r}")
        return 1

    n_pass = 0
    print(f"\nRunning {len(cases)} tool-eval case(s)...\n" + "=" * 68)
    for c in cases:
        passed, checks, info, dt = run_case(c)
        n_pass += passed
        head = "PASS" if passed else "FAIL"
        print(f"\n[{head}] {c['name']}   ({dt:.1f}s)")
        print(f"       {info}")
        for name, ok, detail in checks:
            print(f"        {'✓' if ok else '✗'} {name:20} {detail}")
    print("\n" + "=" * 68)
    print(f"SUMMARY: {n_pass}/{len(cases)} cases passed")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
