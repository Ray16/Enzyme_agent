import unittest
from enzyme_discovery_agent.models.expression import assess_expression, expression_preference


class ExpressionEvidenceTest(unittest.TestCase):
    def test_native_host_and_origin_are_not_recombinant_expression_evidence(self):
        for organism, lineage in [("Escherichia coli", ["Bacteria"]), ("Synthetic plant", ["Eukaryota"])]:
            value = assess_expression("P12345", {"organism": {"scientificName": organism, "lineage": lineage}})
            self.assertFalse(value["ecoli_expression"])
            self.assertFalse(value["bacterial_expression"])
            self.assertEqual(value["preferred_host_assessment"]["level"], "unknown")
        self.assertEqual(expression_preference({"expression_data": {"ecoli_expression": True}}), 1)

    def test_function_comment_can_document_heterologous_activity(self):
        entry = {"comments": [{"commentType": "FUNCTION", "texts": [{
            "value": "Catalyzes the native reaction when expressed in a heterologous system (E.coli)."}]}]}
        value = assess_expression("P12345", entry)
        self.assertEqual(value["preferred_host_assessment"]["level"], "reported_active")
        self.assertIn("heterologous", value["preferred_host_assessment"]["evidence"][0]["quote"])
        self.assertTrue(value["ecoli_expression"])

    def test_irrelevant_host_mentions_and_failed_expression_are_not_successes(self):
        entry = {"comments": [{"texts": [{"value": "The native protein inhibits Escherichia coli."}]}]}
        self.assertFalse(assess_expression("P12345", entry)["ecoli_expression"])
        entry["comments"][0]["texts"][0]["value"] = "Protein expressed in Escherichia coli was insoluble."
        value = assess_expression("P12345", entry)
        self.assertEqual(value["preferred_host_assessment"]["level"], "expression_problem_reported")
        self.assertEqual(expression_preference({"expression_data": value}), 0)

    def test_only_candidate_matched_polymer_entity_host_can_support_expression(self):
        def fetch(url):
            if "/entry/" in url:
                return {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1", "2"]},
                        "irrelevant_source_organism": "Escherichia coli"}
            if url.endswith("/1"):
                return {"rcsb_polymer_entity_container_identifiers": {"uniprot_ids": ["Q12345"]},
                        "rcsb_entity_host_organism": [{"ncbi_scientific_name": "Escherichia coli"}]}
            return {"rcsb_polymer_entity_container_identifiers": {"uniprot_ids": ["P12345"]},
                    "rcsb_entity_source_organism": [{"ncbi_scientific_name": "Escherichia coli"}]}
        value = assess_expression("P12345", {}, ["1XYZ"], fetch)
        self.assertFalse(value["ecoli_expression"])
        value = assess_expression("Q12345", {}, ["1XYZ"], fetch)
        self.assertEqual(value["preferred_host_assessment"]["level"], "structural_construct_expressed")
        self.assertIn("construct", value["preferred_host_assessment"]["evidence"][0]["construct_scope"])

    def test_sequence_annotations_flag_review_without_claiming_failure(self):
        value = assess_expression("P12345", {"features": [{"type": "Transmembrane"}, {"type": "Signal"}]})
        self.assertEqual(value["preferred_host_assessment"]["level"], "unknown")
        self.assertEqual(len(value["preferred_host_assessment"]["sequence_annotation_risks"]), 2)


if __name__ == "__main__":
    unittest.main()
