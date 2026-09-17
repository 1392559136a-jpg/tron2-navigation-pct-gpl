import pathlib
import sys

import numpy as np


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pct_goal_editor import PCTGoalEditor


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warning(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self):
        self.logger = FakeLogger()

    def get_logger(self):
        return self.logger


class FakeLayerIndex:
    def __init__(self, layers):
        self.layers = layers

    def find_available_layers(self, _position):
        return self.layers


def make_editor(layers):
    editor = PCTGoalEditor.__new__(PCTGoalEditor)
    editor.node = FakeNode()
    editor.index = FakeLayerIndex(layers)
    editor.z_alignment = 0.0
    editor.path_height_offset = 0.5
    return editor


def test_projects_body_pose_to_closest_expected_floor_without_changing_xy():
    editor = make_editor(
        [
            {
                "layer_idx": 0,
                "elevation": -2.8,
                "spatial_distance": 0.1,
                "traversability_cost": 0.1,
            },
            {
                "layer_idx": 1,
                "elevation": -0.85,
                "spatial_distance": 0.2,
                "traversability_cost": 0.2,
            },
        ]
    )

    ground, layer = editor.project_start_to_ground(
        np.array([1.2, -3.4, -0.03]), log=False
    )

    np.testing.assert_allclose(ground, [1.2, -3.4, -0.85])
    assert layer == 1
    assert editor.node.logger.messages == []


def test_returns_body_pose_when_no_floor_is_available():
    editor = make_editor([])
    body = np.array([1.0, 2.0, 3.0])

    ground, layer = editor.project_start_to_ground(body, log=False)

    np.testing.assert_allclose(ground, body)
    assert layer == -1