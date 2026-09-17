#!/usr/bin/env python3
"""Regression test for tomogram states that share a physical elevation."""

import sys
import unittest
from pathlib import Path

import numpy as np


PLANNER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLANNER_ROOT))

from lib import a_star  # noqa: E402


class MultilayerAstarStateTest(unittest.TestCase):
    def make_planner(self):
        num_layers = 2
        rows = 3
        cols = 3
        cost = np.zeros((num_layers * rows, cols), dtype=np.float64)
        # Both layers deliberately have identical physical elevations. Layer
        # identity must still remain part of the A* state key and goal check.
        height = np.zeros_like(cost)
        gateway = np.zeros_like(cost)
        gateway[:rows] = 1.0
        gateway[rows:] = -1.0

        planner = a_star.Astar(a_star.HeuristicType.DIAGONAL)
        planner.init(1.0, num_layers, 1.0, 0.0, cost, height, gateway)
        return planner

    def assert_crosses_layer(self, start_layer, goal_layer):
        planner = self.make_planner()
        start = np.asarray([start_layer, 1, 1], dtype=np.int32)
        goal = np.asarray([goal_layer, 1, 1], dtype=np.int32)

        self.assertTrue(planner.search(start, goal))
        path = planner.get_result_matrix()
        self.assertGreater(len(path), 0)
        layers = np.concatenate(([start_layer], path[:, 0].astype(int)))
        self.assertEqual(layers[-1], goal_layer)
        self.assertTrue(np.any(np.diff(layers) != 0))
        self.assertTrue(np.all(np.abs(np.diff(layers)) <= 1))

    def test_upward_same_elevation_states_are_distinct(self):
        self.assert_crosses_layer(0, 1)

    def test_downward_same_elevation_states_are_distinct(self):
        self.assert_crosses_layer(1, 0)


if __name__ == '__main__':
    unittest.main()