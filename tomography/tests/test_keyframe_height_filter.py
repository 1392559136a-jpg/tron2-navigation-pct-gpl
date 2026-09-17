import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from extract_mros_keyframe_map import filter_relative_height  # noqa: E402


class KeyframeHeightFilterTest(unittest.TestCase):
    def test_keeps_support_surfaces_and_removes_overhead_returns(self):
        points = np.array(
            [
                [0.0, 0.0, -0.8],
                [1.0, 0.0, 0.2],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.01],
                [4.0, 0.0, 2.3],
            ],
            dtype=np.float32,
        )

        filtered = filter_relative_height(points, maximum_z=1.0)

        np.testing.assert_array_equal(filtered, points[:3])
        self.assertEqual(points.shape, (5, 3))

    def test_rejects_inverted_height_range(self):
        with self.assertRaisesRegex(ValueError, 'must not exceed'):
            filter_relative_height(
                np.zeros((1, 3), dtype=np.float32),
                minimum_z=1.0,
                maximum_z=-1.0,
            )


if __name__ == '__main__':
    unittest.main()