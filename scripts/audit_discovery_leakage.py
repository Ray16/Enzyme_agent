"""Offline source audit. Never imported by the discovery runtime."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def audit(project):
    package = project / "enzyme_discovery_agent"
    benchmark_tree = ast.parse((package / "eval" / "benchmarks.py").read_text())
    benchmarks = next(ast.literal_eval(node.value) for node in benchmark_tree.body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BENCHMARKS" for t in node.targets))
    answer_terms = {str(term).lower() for benchmark in benchmarks.values()
                    for group in benchmark.get("primary_targets", [])
                    for term in benchmark.get("recall_targets", {}).get(group, {}).get("values", [])}
    rules = (package / "CLAUDE.md").read_text()
    # Specific products/identifiers and adjacent hints from the standing rules.
    # Classification enums are allowed separately from search/prompt examples.
    rule_terms = set(re.findall(r"`([^`]+)`", rules.split("**What IS allowed")[0]))
    rule_patterns = {term: re.compile(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)")
                     for term in rule_terms if " " not in term and "/" not in term}
    answer_patterns = {term: re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)") for term in answer_terms}
    excluded = {"eval", "tests", "runs", "figures", "slides", "__pycache__"}
    classification = {"chemistry_logic.py", "static/ec_bond_map.py", "tools/substrate.py", "models/target_reasoning.py"}
    findings, symbols, hashes, imports = [], {}, {}, []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if excluded & set(relative.parts):
            continue
        source = path.read_text()
        name = str(relative)
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
        low = source.lower()
        for term, pattern in answer_patterns.items():
            if pattern.search(low):
                findings.append({"file": name, "kind": "answer_literal", "term": term})
        for term, pattern in rule_patterns.items():
            if pattern.search(low) and name not in classification:
                # Explicitly allowed acronym rejection lists do not hint at an
                # enzyme class; review other occurrences individually.
                if term.upper() in {"NRPS", "ATP"}:
                    continue
                findings.append({"file": name, "kind": "instruction_literal", "term": term})
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for symbol in re.findall(r"\b[A-Z][a-z]{1,3}[A-Z0-9]\b", node.value):
                    symbols.setdefault(symbol, set()).add(name)
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if set(module.split(".")) & {"eval", "tests", "analysis", "figures", "runs"}:
                    findings.append({"file": name, "kind": "forbidden_namespace_import", "module": module})
                imports.append({"file": name, "module": module, "relative_level": node.level})
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("enzyme_discovery_agent.eval"):
                        findings.append({"file": name, "kind": "forbidden_namespace_import", "module": alias.name})
    return {"runtime_files_scanned": len(hashes), "findings": findings,
            "short_mixed_case_literals_for_review": {k: sorted(v) for k, v in symbols.items()},
            "classification_vocabularies": sorted(classification),
            "excluded_namespaces": sorted(excluded), "source_sha256": hashes,
            "import_edges": imports,
            "limits": ["Static checks cannot establish biological generalization or prove absence of every possible hint.",
                       "Historical figure scripts are presentation artifacts, not discovery inputs.",
                       "Development diagnostics are separate from held-out evaluation."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("runtime_files_scanned", "findings", "short_mixed_case_literals_for_review")}, indent=2))
    return bool(result["findings"])


if __name__ == "__main__":
    raise SystemExit(main())
