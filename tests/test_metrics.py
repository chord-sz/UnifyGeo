import unittest

import numpy as np
import torch

from unifygeo.evaluation.vigor import (
    apply_reranking,
    gated_lf_cvgl_metrics,
    official_hit_rate_flags,
    retrieval_metrics,
)


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        metrics = retrieval_metrics(
            np.array([1, 2, 10, 11]),
            np.array([True, True, True, False]),
            np.array([True, True, False, False]),
            reference_count=100,
        )
        self.assertEqual(metrics["R@1"], 25.0)
        self.assertEqual(metrics["R@5"], 50.0)
        self.assertEqual(metrics["R@10"], 75.0)
        self.assertEqual(metrics["Hit_Rate"], 75.0)

    def test_semi_positives_are_masked_for_hit_rate(self):
        flags = official_hit_rate_flags(
            np.array([[11, 12, 10, 90, 91], [21, 90, 20, 22, 23]]),
            np.array([[1.0, 0.9, 0.8, 0.7, 0.6], [1.0, 0.9, 0.8, 0.7, 0.6]]),
            np.array([[10, 11, 12, 13], [20, 21, 22, 23]]),
        )
        self.assertEqual(flags.tolist(), [True, False])

    def test_reranking_uses_the_same_hit_rate_rule(self):
        ranking = {
            "ranks": torch.tensor([2, 2, 7]),
            "positive_indices": torch.tensor([
                [10, 11, 12, 13],
                [20, 21, 22, 23],
                [30, 31, 32, 33],
            ]),
            "topk_indices": torch.tensor([
                [11, 10, 90, 91, 92],
                [90, 20, 21, 22, 23],
                [90, 91, 92, 93, 94],
            ]),
            "topk_scores": torch.zeros(3, 5),
            "paper_hit": torch.tensor([True, False, False]),
            "top1_any_positive": torch.tensor([True, False, False]),
        }
        local_scores = torch.tensor([
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        reranked = apply_reranking(ranking, local_scores)
        self.assertEqual(reranked["post_ranks"].tolist(), [1, 1, 7])
        self.assertEqual(reranked["post_paper_hit"].tolist(), [True, True, False])

    def test_lf_cvgl_uses_all_queries_as_the_denominator(self):
        metrics = gated_lf_cvgl_metrics(
            np.array([0.5, 0.5, 2.0, 0.5]),
            np.array([True, False, True, True]),
        )
        self.assertEqual(metrics["accuracy_percent"]["1m"], 50.0)


if __name__ == "__main__":
    unittest.main()
