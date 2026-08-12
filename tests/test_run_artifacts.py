import unittest
from pathlib import Path
from uuid import uuid4

from bench.runner import _make_run_paths


class RunArtifactsTest(unittest.TestCase):
    def test_make_run_paths_creates_immutable_run_dir(self):
        root = Path("data/test_runs")
        run_id = f"unit_test_run_{uuid4().hex}"
        run_dir = root / run_id

        actual_id, actual_dir = _make_run_paths(
            max_papers=2,
            no_llm=True,
            run_id=run_id,
            output_root=str(root),
        )

        self.assertEqual(actual_id, run_id)
        self.assertTrue(actual_dir.exists())
        self.assertEqual(actual_dir, run_dir)


if __name__ == "__main__":
    unittest.main()
