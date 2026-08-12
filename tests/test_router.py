import unittest

from paperpilot.core.router import route_query


class RouterTest(unittest.TestCase):
    def test_numerical_query_prefers_table_and_numeric_policy(self):
        plan = route_query("Which model has the highest F1 score?")

        self.assertEqual(plan.query_type, "numerical")
        self.assertEqual(plan.answer_style, "numeric_with_units")
        self.assertIn("table", plan.source_preference)
        self.assertIn("must_find_numeric_value", plan.confidence_checks)

    def test_comparison_query_requires_two_sided_evidence(self):
        plan = route_query("How does BERT compare to ELMo?")

        self.assertEqual(plan.query_type, "comparison")
        self.assertEqual(plan.answer_style, "structured_comparison")
        self.assertIn("require_two_sided_evidence", plan.confidence_checks)

    def test_simple_factual_query_uses_standard_strategy(self):
        plan = route_query("What dataset does the paper use?")

        self.assertEqual(plan.query_type, "factual")
        self.assertEqual(plan.route_level, "small_router")
        self.assertEqual(plan.retrieval_strategy, "standard")
        self.assertEqual(plan.rewrite_queries, [])

    def test_complex_why_query_uses_fusion_strategy(self):
        plan = route_query("Why does the method improve robustness?")

        self.assertEqual(plan.query_type, "why_how")
        self.assertEqual(plan.route_level, "small_router")
        self.assertEqual(plan.retrieval_strategy, "fusion")
        self.assertGreater(plan.complexity_score, 0.5)


if __name__ == "__main__":
    unittest.main()
