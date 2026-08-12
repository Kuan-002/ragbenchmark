import unittest
from unittest.mock import patch

from bench import retrievers


class MultiQueryFusionTest(unittest.TestCase):
    def test_query_variants_expand_numerical_intent(self):
        variants = retrievers.query_variants(
            "Which model has the highest F1 score?",
            {
                "query_type": "numerical",
                "comparison_entities": [],
            },
        )

        self.assertEqual(variants[0], "Which model has the highest F1 score?")
        self.assertTrue(any("table results metrics" in v for v in variants))
        self.assertGreaterEqual(len(variants), 3)

    def test_query_variants_do_not_expand_simple_factual_intent(self):
        variants = retrievers.query_variants(
            "What dataset does the paper use?",
            {"query_type": "factual"},
        )

        self.assertEqual(variants, ["What dataset does the paper use?"])

    def test_multi_query_rrf_recovers_candidate_from_rewrite(self):
        calls = []

        def fake_rrf(query_text, _bm25, _dense, top_k=5,
                     rrf_k=60, candidate_k=100):
            calls.append(query_text)
            if "table results metrics" in query_text:
                return [("gold", 1.0), ("distractor", 0.5)]
            return [("distractor", 1.0)]

        with patch("bench.retrievers.retrieve_rrf", side_effect=fake_rrf):
            rows = retrievers.retrieve_multi_query_rrf(
                "Which model has the highest F1 score?",
                None,
                None,
                query_plan={"query_type": "numerical"},
                top_k=3,
            )

        ids = [cid for cid, _score in rows]
        self.assertIn("gold", ids)
        self.assertGreaterEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
