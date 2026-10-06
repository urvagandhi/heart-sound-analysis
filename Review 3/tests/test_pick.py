"""Unit test for pick_explainable_indices in Review 3/utils.py.
Synthetic test requiring no external data files.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure Review 3 is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import pick_explainable_indices


import unittest


class TestPickExplainableIndices(unittest.TestCase):
    def test_pick_explainable_indices_synthetic(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        preds  = np.array([0, 0, 1, 1, 1, 0])
        probs  = np.array([
            [0.95, 0.05],  # 0: correct Normal (high conf)
            [0.70, 0.30],  # 1: correct Normal (lower conf)
            [0.20, 0.80],  # 2: FP (true 0, pred 1)
            [0.10, 0.90],  # 3: correct Abnormal (high conf)
            [0.35, 0.65],  # 4: correct Abnormal (lower conf)
            [0.85, 0.15],  # 5: FN (true 1, pred 0)
        ])

        metrics = {
            "labels": labels,
            "preds": preds,
            "probs": probs,
        }

        picks = pick_explainable_indices(metrics, n_correct=2, n_wrong=2, rng_seed=42)

        self.assertEqual(len(picks), 4, f"Expected 4 picks, got {len(picks)}")
        self.assertEqual(picks[0], 0, f"Expected index 0 (top correct normal), got {picks[0]}")
        self.assertEqual(picks[1], 3, f"Expected index 3 (top correct abnormal), got {picks[1]}")

        error_picks = picks[2:]
        self.assertIn(2, error_picks, f"Expected False Positive index 2 in error picks, got {error_picks}")
        self.assertIn(5, error_picks, f"Expected False Negative index 5 in error picks, got {error_picks}")


if __name__ == "__main__":
    unittest.main()
