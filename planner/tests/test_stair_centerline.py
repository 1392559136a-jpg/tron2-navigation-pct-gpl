#!/usr/bin/env python3
"""Unit tests for validated straight staircase reference generation."""

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from stair_centerline import (  # noqa: E402
    StairCenterlineError,
    build_stair_centerline,
)


class StairCenterlineTest(unittest.TestCase):
    def make_stair_map(self):
        resolution = 0.25
        center = np.array([0.0, 0.0], dtype=np.float64)
        shape = (4, 21, 21)
        traversability = np.full(shape, np.inf, dtype=np.float32)
        elevation = np.full(shape, np.nan, dtype=np.float32)
        start = np.array([-2.0, 0.0, 0.5], dtype=np.float64)
        goal = np.array([2.0, 0.0, 3.5], dtype=np.float64)
        ratios = np.linspace(0.0, 1.0, 17)
        layers = np.floor(3.0 * ratios + 1.0e-9).astype(np.int32)
        body_z = start[2] + ratios * (goal[2] - start[2])
        offset = np.array(shape[1:]) // 2

        for ratio, layer, support_z in zip(ratios, layers, body_z):
            xy = start[:2] + ratio * (goal[:2] - start[:2])
            x_index, y_index = (
                np.rint((xy - center) / resolution).astype(np.int32) + offset
            )
            traversability[layer, x_index, y_index] = 1.0
            elevation[layer, x_index, y_index] = support_z - 0.5

        return {
            'start': start,
            'goal': goal,
            'traversability': traversability,
            'elevation': elevation,
            'center': center,
            'resolution': resolution,
            'layers': layers,
        }

    def build(self, data, **overrides):
        arguments = {
            'start_pos': data['start'],
            'goal_pos': data['goal'],
            'traversability': data['traversability'],
            'ground_elevation': data['elevation'],
            'center': data['center'],
            'resolution': data['resolution'],
            'start_layer': int(data['layers'][0]),
        }
        arguments.update(overrides)
        return build_stair_centerline(**arguments)

    def test_builds_complete_xy_straight_monotonic_path(self):
        data = self.make_stair_map()
        result = self.build(data)

        self.assertEqual(result.trajectory.shape, (17, 3))
        np.testing.assert_allclose(result.trajectory[0], data['start'])
        np.testing.assert_allclose(result.trajectory[-1], data['goal'])
        direction = data['goal'][:2] - data['start'][:2]
        relative = result.trajectory[:, :2] - data['start'][:2]
        cross_track = np.abs(
            direction[0] * relative[:, 1] - direction[1] * relative[:, 0]
        ) / np.linalg.norm(direction)
        self.assertLessEqual(float(np.max(cross_track)), 1.0e-12)
        self.assertTrue(np.all(np.diff(result.trajectory[:, 2]) >= -1.0e-12))
        self.assertTrue(np.all(np.diff(result.layers) >= 0))
        self.assertLessEqual(int(np.max(np.abs(np.diff(result.layers)))), 1)
        self.assertLessEqual(result.max_cost, 20.0)
        self.assertLessEqual(result.max_support_error, 0.25)

    def test_descending_path_keeps_layers_and_height_monotonic(self):
        data = self.make_stair_map()
        data['start'], data['goal'] = data['goal'], data['start']
        data['layers'] = data['layers'][::-1]
        result = self.build(data)

        self.assertTrue(np.all(np.diff(result.trajectory[:, 2]) <= 1.0e-12))
        self.assertTrue(np.all(np.diff(result.layers) <= 0))
        self.assertEqual(int(result.layers[0]), 3)
        self.assertEqual(int(result.layers[-1]), 0)

    def test_rejects_missing_traversable_support(self):
        data = self.make_stair_map()
        middle = len(data['layers']) // 2
        x_index = 2 + middle
        data['traversability'][:, x_index, 10] = 25.0

        with self.assertRaisesRegex(
            StairCenterlineError, 'No traversable PCT support'
        ):
            self.build(data)

    def test_rejects_non_adjacent_layer_jump(self):
        data = self.make_stair_map()
        second_x_index = 3
        data['traversability'][:, second_x_index, 10] = np.inf
        data['elevation'][:, second_x_index, 10] = np.nan
        data['traversability'][2, second_x_index, 10] = 1.0
        data['elevation'][2, second_x_index, 10] = 0.1875

        with self.assertRaisesRegex(
            StairCenterlineError, 'layer continuity failed'
        ):
            self.build(data)

    def test_rejects_sudden_support_height_jump(self):
        data = self.make_stair_map()
        middle = len(data['layers']) // 2
        x_index = 2 + middle
        layer = int(data['layers'][middle])
        data['elevation'][layer, x_index, 10] += 0.6

        with self.assertRaisesRegex(
            StairCenterlineError, 'support height jumps'
        ):
            self.build(data)

    def test_rejects_excessive_smoothed_support_error(self):
        data = self.make_stair_map()
        with self.assertRaisesRegex(
            StairCenterlineError, 'support error'
        ):
            self.build(
                data,
                max_support_error=0.01,
                max_support_step=1.0,
            )

    def test_rejects_non_stair_vertical_span(self):
        data = self.make_stair_map()
        data['goal'] = data['goal'].copy()
        data['goal'][2] = data['start'][2] + 0.1

        with self.assertRaisesRegex(
            StairCenterlineError, 'vertical span'
        ):
            self.build(data)

    def test_rejects_out_of_bounds_path(self):
        data = self.make_stair_map()
        data['goal'] = np.array([20.0, 0.0, 3.5], dtype=np.float64)

        with self.assertRaisesRegex(
            StairCenterlineError, 'outside tomogram bounds'
        ):
            self.build(data)


if __name__ == '__main__':
    unittest.main()
