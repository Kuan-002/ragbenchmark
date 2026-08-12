import unittest

from bench.metrics import (
    all_evidence_recall_at_k,
    first_gold_rank,
    hit_at_k,
    mrr,
    recall_at_k,
)


class MetricsTest(unittest.TestCase):
    def test_single_gold_retrieval_metrics(self):
        retrieved = ["a", "gold", "c"]
        gold = ["gold"]

        self.assertEqual(hit_at_k(retrieved, gold, 1), 0.0)
        self.assertEqual(hit_at_k(retrieved, gold, 3), 1.0)
        self.assertEqual(first_gold_rank(retrieved, gold), 2)
        self.assertAlmostEqual(mrr(retrieved, gold), 0.5)

    def test_multi_gold_recall_metrics(self):
        retrieved = ["g1", "x", "g2", "y", "z"]
        gold = ["g1", "g2", "g3"]

        self.assertAlmostEqual(recall_at_k(retrieved, gold, 5), 2 / 3)
        self.assertEqual(all_evidence_recall_at_k(retrieved, gold, 5), 0.0)
        self.assertEqual(all_evidence_recall_at_k(["g1", "g2", "g3"], gold, 3), 1.0)


if __name__ == "__main__":
    unittest.main()
