"""The ranked CSV must have one row per accession.

An accession evaluated across several iterations yields several identical
evaluation artifacts; the CSV is a decision list (one representative accession
per ranking row), so those must collapse to a single row.  This is distinct from
ortholog collapse (which merges *different* accessions) -- here the accession is
the same.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.output.csv_generator import generate_candidates_csv


def _eval(acc, name):
    return {"status": "success", "data": {
        "accession": acc, "protein_name": name, "gene_names": [name.lower()],
        "organism": "E. coli", "ec_numbers": ["6.3.1.1"], "interpro_family": "Some family",
    }}


class TestCsvDedup(unittest.TestCase):
    def test_same_accession_collapses_to_one_row(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "pass1_ranking.json").write_text(json.dumps([
                {"rank": 1, "accession": "P00001", "tier": 1, "tier_label": "strong lead",
                 "substrate_fit": 0.5, "evidence_level": "high", "family_relevance": "on-target",
                 "direction": "formation"},
                {"rank": 2, "accession": "P00002", "tier": 2, "tier_label": "worth testing",
                 "substrate_fit": 0.3, "evidence_level": "low", "family_relevance": "on-target",
                 "direction": "formation"},
            ]))
            # P00001 evaluated 3x, P00002 twice -> 5 eval entries, 2 accessions.
            evals = [_eval("P00001", "Alpha"), _eval("P00001", "Alpha"), _eval("P00001", "Alpha"),
                     _eval("P00002", "Beta"), _eval("P00002", "Beta")]
            generate_candidates_csv(run_dir, evals)

            rows = list(csv.DictReader(open(run_dir / "ranked_candidates.csv")))
            accs = [r["accession"] for r in rows]
            self.assertEqual(len(rows), 2, f"expected one row per accession, got {accs}")
            self.assertEqual(accs, ["P00001", "P00002"])


if __name__ == "__main__":
    unittest.main()
