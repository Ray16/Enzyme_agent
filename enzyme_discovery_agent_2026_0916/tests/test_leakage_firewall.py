"""Data-leakage firewall: the agent must never see the benchmark answers.

The eval benchmarks name the specific answer enzymes (DdaG/SfaB, accessions,
NylC...) purely to SCORE a run after the fact. Those identifiers must never
appear in the agent's own runtime (prompts, tools, gates, loop), and the agent
must never import ``eval`` — otherwise the search would be seeded with its own
answer and the recall numbers would be meaningless.

Only the PRIMARY (specific) target values are forbidden; generic mechanism-class
vocabulary (e.g. "amide ligase", "atp-grasp") is legitimate agent reasoning and
is NOT restricted.
"""

import re
import unittest
from pathlib import Path

from enzyme_discovery_agent.eval.benchmarks import BENCHMARKS

_PKG = Path(__file__).resolve().parent.parent          # enzyme_discovery_agent/
# The firewall protects the agent's RUNTIME (prompts, tools, gates, loop): those
# must not name the answer, or the search would be seeded.  ``/eval/`` is the
# scoring oracle and ``/tests/`` are tests; ``/figures/`` and ``/runs/`` are
# post-hoc artefacts that VISUALISE or RECORD a completed run's results (which of
# course include the discovered families) and are never read during a search, so
# they are not runtime and are excluded from the scan.
_EXCLUDE = ("/eval/", "/tests/", "/figures/", "/runs/")


def _forbidden_terms() -> set[str]:
    terms: set[str] = set()
    for b in BENCHMARKS.values():
        targets = b.get("recall_targets", {})
        for name in (b.get("primary_targets") or []):
            for v in targets.get(name, {}).get("values", []):
                terms.add(v.lower())
    return terms


def _runtime_files():
    for p in _PKG.rglob("*.py"):
        if not any(x in str(p) for x in _EXCLUDE):
            yield p


class TestLeakageFirewall(unittest.TestCase):
    def test_no_answer_terms_in_agent_runtime(self):
        forbidden = _forbidden_terms()
        self.assertTrue(forbidden, "no forbidden terms resolved — check benchmarks")
        # Match answer terms as whole tokens, NOT substrings: a short gene marker
        # like "nyla" must not false-positive inside legitimate vocabulary such as
        # "ade-NYLA-tion". Real leakage is the gene/accession as its own word.
        patterns = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in forbidden}
        offenders = []
        for p in _runtime_files():
            text = p.read_text(errors="ignore").lower()
            offenders += [(p.name, t) for t, rx in patterns.items() if rx.search(text)]
        self.assertEqual(offenders, [], f"answer terms leaked into agent runtime: {offenders}")

    def test_agent_runtime_does_not_import_eval(self):
        pat = re.compile(r"enzyme_discovery_agent\.eval|from\s+\.\.?eval\b|import\s+eval\b")
        offenders = [p.name for p in _runtime_files()
                     if pat.search(p.read_text(errors="ignore"))]
        self.assertEqual(offenders, [], f"agent runtime imports eval: {offenders}")


if __name__ == "__main__":
    unittest.main()
