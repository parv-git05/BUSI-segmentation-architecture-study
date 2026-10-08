"""
Unit tests for Segmentation Metrics Suite (Dice, IoU, Precision, Recall, Specificity, HD95, NSD).
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add 00_Common directory to sys.path
common_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from metrics.segmentation_metrics import (
    compute_dice,
    compute_iou,
    compute_precision,
    compute_recall,
    compute_specificity,
    compute_hd95,
    compute_nsd,
    calculate_sample_metrics,
    aggregate_metrics
)


class TestSegmentationMetrics(unittest.TestCase):

    def setUp(self):
        # 100x100 synthetic grids
        self.shape = (100, 100)

        # Perfect Match Case
        self.gt_full = np.zeros(self.shape, dtype=bool)
        self.gt_full[20:60, 20:60] = True
        self.pred_full = self.gt_full.copy()

        # Partial Overlap Case
        # GT: 40x40 square = 1600 px
        # Pred: 40x40 square shifted by 10 px -> overlap = 30x40 = 1200 px
        self.pred_partial = np.zeros(self.shape, dtype=bool)
        self.pred_partial[20:60, 30:70] = True

        # Disjoint Case
        self.pred_disjoint = np.zeros(self.shape, dtype=bool)
        self.pred_disjoint[70:90, 70:90] = True

        # Both Empty Case
        self.empty_gt = np.zeros(self.shape, dtype=bool)
        self.empty_pred = np.zeros(self.shape, dtype=bool)

    def test_perfect_match_metrics(self):
        """Verifies perfect overlap yields maximum metric scores."""
        m = calculate_sample_metrics(self.pred_full, self.gt_full)
        self.assertAlmostEqual(m["dice"], 1.0)
        self.assertAlmostEqual(m["iou"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["specificity"], 1.0)
        self.assertAlmostEqual(m["hd95"], 0.0)
        self.assertAlmostEqual(m["nsd_1px"], 1.0)

    def test_partial_overlap_metrics(self):
        """Verifies partial overlap mathematical values."""
        # TP = 1200, FP = 400, FN = 400
        # Dice = 2 * 1200 / (2*1200 + 400 + 400) = 2400 / 3200 = 0.75
        # IoU = 1200 / (1200 + 400 + 400) = 1200 / 2000 = 0.60
        m = calculate_sample_metrics(self.pred_partial, self.gt_full)
        self.assertAlmostEqual(m["dice"], 0.75)
        self.assertAlmostEqual(m["iou"], 0.60)
        self.assertAlmostEqual(m["precision"], 0.75)
        self.assertAlmostEqual(m["recall"], 0.75)
        self.assertGreater(m["hd95"], 0.0)

    def test_both_empty_edge_case(self):
        """Verifies edge case where both GT and Prediction are empty."""
        m = calculate_sample_metrics(self.empty_pred, self.empty_gt)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["hd95"], 0.0)
        self.assertEqual(m["nsd_1px"], 1.0)

    def test_one_empty_edge_case(self):
        """Verifies edge case where GT has lesion but Prediction is empty."""
        m = calculate_sample_metrics(self.empty_pred, self.gt_full)
        self.assertEqual(m["dice"], 0.0)
        self.assertEqual(m["iou"], 0.0)
        self.assertTrue(np.isnan(m["hd95"]))
        self.assertEqual(m["nsd_1px"], 0.0)

    def test_aggregation(self):
        """Verifies aggregation returns valid means and handles NaNs safely."""
        m1 = calculate_sample_metrics(self.pred_full, self.gt_full)
        m2 = calculate_sample_metrics(self.empty_pred, self.gt_full)  # HD95 is NaN

        agg = aggregate_metrics([m1, m2])
        self.assertIn("dice_mean", agg)
        self.assertEqual(agg["dice_mean"], 0.5)
        self.assertEqual(agg["hd95_valid_count"], 1)
        self.assertEqual(agg["hd95_mean"], 0.0)  # Only 1 valid sample (0.0)


if __name__ == "__main__":
    unittest.main()
