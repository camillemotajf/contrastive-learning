import unittest

import numpy as np

from src.experiments._common import _balanced_subsample_indices


class BalancedSubsampleTests(unittest.TestCase):
    def test_balanced_subsample_takes_equal_count_per_class(self):
        y = np.array([0] * 90 + [1] * 10)

        idx = _balanced_subsample_indices(y, n=20, seed=42)

        sampled = y[idx]
        self.assertEqual(20, len(sampled))
        self.assertEqual(10, int((sampled == 0).sum()))
        self.assertEqual(10, int((sampled == 1).sum()))

    def test_balanced_subsample_is_capped_by_smallest_class(self):
        y = np.array([0] * 90 + [1] * 10)

        idx = _balanced_subsample_indices(y, n=80, seed=42)

        sampled = y[idx]
        self.assertEqual(20, len(sampled))
        self.assertEqual(10, int((sampled == 0).sum()))
        self.assertEqual(10, int((sampled == 1).sum()))


if __name__ == "__main__":
    unittest.main()
