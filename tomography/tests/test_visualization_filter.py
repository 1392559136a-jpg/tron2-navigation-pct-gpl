import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from publish_tomogram_from_pickle import (  # noqa: E402
    build_visualization_points,
    remove_small_components,
)
from pct_goal_editor import PointCloudLayerIndex  # noqa: E402


class VisualizationFilterTest(unittest.TestCase):
    def test_remove_small_components_uses_eight_connectivity(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        mask[5, 5] = True

        filtered = remove_small_components(mask, minimum_cells=2)

        self.assertTrue(filtered[0, 0])
        self.assertTrue(filtered[1, 1])
        self.assertFalse(filtered[5, 5])

    def test_build_points_filters_cost_height_and_small_islands(self):
        data = np.full((5, 1, 8, 8), np.nan, dtype=np.float32)
        ground = data[3, 0]
        cost = data[0, 0]

        main_floor = ((1, 1), (1, 2), (2, 1), (2, 2))
        roof = ((1, 5), (1, 6), (2, 5), (2, 6))
        high_cost = ((5, 1), (5, 2), (6, 1), (6, 2))
        for x_idx, y_idx in main_floor:
            ground[x_idx, y_idx] = 0.0
            cost[x_idx, y_idx] = 1.0
        ground[4, 4] = 0.0
        cost[4, 4] = 1.0
        for x_idx, y_idx in roof:
            ground[x_idx, y_idx] = 3.0
            cost[x_idx, y_idx] = 1.0
        for x_idx, y_idx in high_cost:
            ground[x_idx, y_idx] = 0.0
            cost[x_idx, y_idx] = 30.0

        points = build_visualization_points(
            data,
            resolution=0.1,
            center=np.zeros(2),
            slice_dh=0.5,
            z_alignment=0.0,
            cost_threshold=20.0,
            minimum_component_cells=3,
            minimum_z=-1.0,
            maximum_z=1.0,
        )

        self.assertEqual(points.shape, (4, 4))
        np.testing.assert_allclose(points[:, 2], 0.0)
        self.assertLessEqual(float(np.max(points[:, 3])), 20.0)

    def test_rejects_invalid_filter_ranges(self):
        data = np.full((5, 1, 1, 1), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'minimum_z'):
            build_visualization_points(
                data,
                resolution=0.1,
                center=np.zeros(2),
                slice_dh=0.5,
                z_alignment=0.0,
                minimum_z=1.0,
                maximum_z=-1.0,
            )

    def test_hide_terminal_envelope_only_changes_visualization(self):
        data = np.full((5, 3, 4, 4), np.nan, dtype=np.float32)
        data[3, 0, 0, 0] = 0.0
        data[0, 0, 0, 0] = 1.0
        data[3, 1, 1, 1] = 1.0
        data[0, 1, 1, 1] = 2.0
        data[3, 2, 2, 2] = 3.0
        data[0, 2, 2, 2] = 3.0
        original = data.copy()

        shown = build_visualization_points(
            data,
            resolution=0.1,
            center=np.zeros(2),
            slice_dh=0.5,
            z_alignment=0.0,
        )
        hidden = build_visualization_points(
            data,
            resolution=0.1,
            center=np.zeros(2),
            slice_dh=0.5,
            z_alignment=0.0,
            hide_terminal_envelope=True,
        )

        self.assertEqual(shown.shape, (3, 4))
        self.assertEqual(hidden.shape, (2, 4))
        self.assertNotIn(3.0, hidden[:, 2])
        np.testing.assert_equal(data, original)

    def test_point_cloud_layer_index_recovers_distinct_floors(self):
        points = np.array(
            [
                [0.00, 0.00, 0.00, 5.0],
                [0.10, 0.00, 0.05, 1.0],
                [0.00, 0.10, 3.00, 2.0],
                [0.05, 0.10, 3.04, 1.0],
                [0.00, 0.00, 6.00, 30.0],
            ],
            dtype=np.float32,
        )
        index = PointCloudLayerIndex(
            points,
            cost_threshold=20.0,
            search_radius=0.5,
            floor_tolerance=0.2,
        )

        layers = index.find_available_layers([0.0, 0.0])

        self.assertEqual(len(layers), 2)
        self.assertEqual([item['layer_idx'] for item in layers], [0, 1])
        self.assertAlmostEqual(layers[0]['elevation'], 0.0)
        self.assertAlmostEqual(layers[1]['elevation'], 3.0)
        self.assertLessEqual(layers[0]['traversability_cost'], 5.0)


if __name__ == '__main__':
    unittest.main()