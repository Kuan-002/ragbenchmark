import unittest

from paperpilot.core.numeric_reasoner import numeric_reasoning_notes


class NumericReasonerTest(unittest.TestCase):
    def test_extracts_table_values_and_computes_maximum(self):
        retrieved = [{
            "score": 1.0,
            "chunk": {
                "chunk_id": "t1",
                "chunk_type": "table",
                "caption": "Main results",
                "section": "Results",
                "text": "| Model | F1 |\n|---|---|\n| Base | 71.2 |\n| Large | 83.5 |",
                "table_columns": ["Model", "F1"],
                "table_rows": [["Base", "71.2"], ["Large", "83.5"]],
            },
        }]

        notes, facts = numeric_reasoning_notes(
            "Which model has the highest F1 score?",
            retrieved,
            {"query_type": "numerical", "needs_calculation": True},
        )

        self.assertEqual(len(facts), 2)
        self.assertIn("Candidate maximum: 83.5", notes)
        self.assertIn("Large / F1", notes)


if __name__ == "__main__":
    unittest.main()
