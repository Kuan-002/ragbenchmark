import unittest

from bench.llm_judge import _json_object, _verdict_from_score


class LlmJudgeTest(unittest.TestCase):
    def test_parses_json_inside_markdown_fence(self):
        data = _json_object('```json\n{"score": 4, "verdict": "FULLY_SUFFICIENT"}\n```')

        self.assertEqual(data["score"], 4)
        self.assertEqual(data["verdict"], "FULLY_SUFFICIENT")

    def test_verdict_mapping(self):
        self.assertEqual(_verdict_from_score(5), "FULLY_SUFFICIENT")
        self.assertEqual(_verdict_from_score(3), "PARTIALLY_SUFFICIENT")
        self.assertEqual(_verdict_from_score(2), "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
