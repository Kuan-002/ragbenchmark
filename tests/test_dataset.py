import unittest

from bench.dataset import extract_answerable_queries


class DatasetTest(unittest.TestCase):
    def test_extracts_reference_answers_and_merges_annotator_evidence(self):
        papers = [{
            "id": "p1",
            "qas": [{
                "question": "What is reported?",
                "question_id": "q1",
                "answers": [
                    {"answer": {
                        "unanswerable": False,
                        "extractive_spans": ["A"],
                        "yes_no": None,
                        "free_form_answer": "",
                        "evidence": ["Evidence A"],
                    }},
                    {"answer": {
                        "unanswerable": False,
                        "extractive_spans": ["B"],
                        "yes_no": None,
                        "free_form_answer": "",
                        "evidence": ["Evidence A", "Evidence B"],
                    }},
                ],
            }],
            "full_text": [],
        }]

        queries = extract_answerable_queries(papers)

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["evidence"], ["Evidence A", "Evidence B"])
        self.assertEqual(queries[0]["reference_answer"], "A | B")


if __name__ == "__main__":
    unittest.main()
