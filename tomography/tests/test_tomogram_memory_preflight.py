import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from build_tomogram_offline import (  # noqa: E402
    GPU_BASE_BUFFER_COUNT,
    calculate_grid_geometry,
    check_gpu_memory,
)
from config.scene_map import SceneMap  # noqa: E402


class TomogramMemoryPreflightTest(unittest.TestCase):
    def test_grid_geometry_counts_six_float32_base_buffers(self):
        scene = SceneMap()
        scene.map.resolution = 0.2
        scene.map.slice_dh = 0.5
        scene.map.ground_h = -1.0
        points = np.array(
            [[0.0, 0.0, -0.5], [2.0, 4.0, 1.0]], dtype=np.float32
        )

        geometry = calculate_grid_geometry(points, scene)

        self.assertEqual(geometry['map_dim_x'], 14)
        self.assertEqual(geometry['map_dim_y'], 24)
        self.assertEqual(geometry['n_slice_init'], 4)
        self.assertEqual(geometry['cells'], 14 * 24 * 4)
        self.assertEqual(
            geometry['base_buffer_bytes'],
            geometry['cells'] * GPU_BASE_BUFFER_COUNT * 4,
        )

    def test_memory_check_accepts_buffers_inside_budget(self):
        geometry = {'base_buffer_bytes': 600, 'map_dim_x': 1,
                    'map_dim_y': 1, 'n_slice_init': 1, 'cells': 1}

        result = check_gpu_memory(
            geometry, memory_fraction=0.65, memory_info=(1000, 1000)
        )

        self.assertEqual(result['safe_budget_bytes'], 650)

    def test_memory_check_refuses_allocation_before_oom(self):
        geometry = {'base_buffer_bytes': 700, 'map_dim_x': 1,
                    'map_dim_y': 1, 'n_slice_init': 1, 'cells': 1}

        with self.assertRaisesRegex(MemoryError, 'refusing allocation'):
            check_gpu_memory(
                geometry, memory_fraction=0.65, memory_info=(1000, 1000)
            )


if __name__ == '__main__':
    unittest.main()
