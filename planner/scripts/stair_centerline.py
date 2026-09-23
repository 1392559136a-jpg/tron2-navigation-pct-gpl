#!/usr/bin/env python3
"""Generate a validated PCT-supported straight staircase reference path."""

from dataclasses import dataclass
import math

import numpy as np


class StairCenterlineError(ValueError):
    """Raised when a straight staircase path is not safe in the tomogram."""


@dataclass
class StairCenterlineResult:
    trajectory: np.ndarray
    layers: np.ndarray
    costs: np.ndarray
    support_body_z: np.ndarray
    length: float
    z_alignment: float
    max_cost: float
    max_support_error: float
    max_support_step: float


def _position(name, value):
    position = np.asarray(value, dtype=np.float64).reshape(-1)
    if position.size < 3 or not np.isfinite(position[:3]).all():
        raise StairCenterlineError(
            '{} must contain finite x, y and z values'.format(name)
        )
    return position[:3].copy()


def _odd_window(requested, sample_count):
    window = min(int(requested), int(sample_count))
    if window < 1:
        raise StairCenterlineError('smoothing_window must be at least 1')
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def build_stair_centerline(
    start_pos,
    goal_pos,
    traversability,
    ground_elevation,
    center,
    resolution,
    path_height_offset=0.5,
    cost_threshold=20.0,
    max_support_error=0.25,
    max_support_step=0.30,
    max_layer_step=1,
    smoothing_window=9,
    minimum_xy_length=0.5,
    minimum_vertical_span=0.5,
    start_layer=None,
    end_layer=None,
):
    """Build one XY-straight body-reference path supported by a PCT tomogram.

    Every sampled XY cell must contain a traversable floor candidate. Selected
    layers must move monotonically through adjacent tomogram layers, and the
    aligned support surface must not contain a sudden vertical discontinuity.
    The output Z profile is smoothed, monotonic, and anchored to the requested
    start and goal body heights.
    """
    start = _position('start_pos', start_pos)
    goal = _position('goal_pos', goal_pos)
    trav = np.asarray(traversability)
    elev = np.asarray(ground_elevation)
    map_center = np.asarray(center, dtype=np.float64).reshape(-1)

    if trav.ndim != 3 or elev.shape != trav.shape:
        raise StairCenterlineError(
            'traversability and ground_elevation must have equal [layer,x,y] shapes'
        )
    if map_center.size < 2 or not np.isfinite(map_center[:2]).all():
        raise StairCenterlineError('center must contain finite x and y values')
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise StairCenterlineError('resolution must be positive and finite')
    path_height_offset = float(path_height_offset)
    cost_threshold = float(cost_threshold)
    max_support_error = float(max_support_error)
    max_support_step = float(max_support_step)
    minimum_xy_length = float(minimum_xy_length)
    minimum_vertical_span = float(minimum_vertical_span)
    max_layer_step = int(max_layer_step)
    if not np.isfinite(path_height_offset):
        raise StairCenterlineError('path_height_offset must be finite')
    if not np.isfinite(cost_threshold) or cost_threshold < 0.0:
        raise StairCenterlineError('cost_threshold must be finite and non-negative')
    if not np.isfinite(max_support_error) or max_support_error <= 0.0:
        raise StairCenterlineError('max_support_error must be positive and finite')
    if not np.isfinite(max_support_step) or max_support_step <= 0.0:
        raise StairCenterlineError('max_support_step must be positive and finite')
    if max_layer_step < 0:
        raise StairCenterlineError('max_layer_step must be non-negative')
    if not np.isfinite(minimum_xy_length) or minimum_xy_length < 0.0:
        raise StairCenterlineError(
            'minimum_xy_length must be finite and non-negative'
        )
    if not np.isfinite(minimum_vertical_span) or minimum_vertical_span < 0.0:
        raise StairCenterlineError(
            'minimum_vertical_span must be finite and non-negative'
        )

    xy_delta = goal[:2] - start[:2]
    length = float(np.linalg.norm(xy_delta))
    if length < minimum_xy_length:
        raise StairCenterlineError(
            'Stair centerline XY length {:.3f}m is below {:.3f}m'.format(
                length, minimum_xy_length
            )
        )
    vertical_span = float(goal[2] - start[2])
    if abs(vertical_span) < minimum_vertical_span:
        raise StairCenterlineError(
            'Stair centerline vertical span {:.3f}m is below {:.3f}m'.format(
                abs(vertical_span), minimum_vertical_span
            )
        )

    sample_count = max(2, int(math.ceil(length / resolution)) + 1)
    ratios = np.linspace(0.0, 1.0, sample_count)
    xy = start[:2] + ratios[:, None] * xy_delta
    expected_body_z = start[2] + ratios * vertical_span
    grid_offset = np.asarray(trav.shape[1:], dtype=np.int64) // 2
    grid_indices = np.rint(
        (xy - map_center[:2]) / resolution
    ).astype(np.int64) + grid_offset
    outside = np.flatnonzero(
        (grid_indices[:, 0] < 0)
        | (grid_indices[:, 0] >= trav.shape[1])
        | (grid_indices[:, 1] < 0)
        | (grid_indices[:, 1] >= trav.shape[2])
    )
    if outside.size:
        sample_index = int(outside[0])
        raise StairCenterlineError(
            'Centerline sample {}/{} at ({:.3f}, {:.3f}) is outside tomogram bounds'.format(
                sample_index + 1,
                sample_count,
                xy[sample_index, 0],
                xy[sample_index, 1],
            )
        )

    def candidates_at(sample_index):
        x_index, y_index = (int(value) for value in grid_indices[sample_index])
        if (
            x_index < 0
            or x_index >= trav.shape[1]
            or y_index < 0
            or y_index >= trav.shape[2]
        ):
            raise StairCenterlineError(
                'Centerline sample {}/{} at ({:.3f}, {:.3f}) is outside tomogram bounds'.format(
                    sample_index + 1,
                    sample_count,
                    xy[sample_index, 0],
                    xy[sample_index, 1],
                )
            )
        cell_costs = np.asarray(trav[:, x_index, y_index], dtype=np.float64)
        cell_heights = np.asarray(elev[:, x_index, y_index], dtype=np.float64)
        valid = np.isfinite(cell_costs) & np.isfinite(cell_heights)
        valid &= cell_costs <= cost_threshold
        valid &= cell_heights > -100.0
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            raise StairCenterlineError(
                'No traversable PCT support at sample {}/{} ({:.3f}, {:.3f}); '
                'cost threshold {:.3f}'.format(
                    sample_index + 1,
                    sample_count,
                    xy[sample_index, 0],
                    xy[sample_index, 1],
                    cost_threshold,
                )
            )
        return x_index, y_index, cell_costs, cell_heights, candidates

    (
        start_x_index,
        start_y_index,
        start_costs,
        start_heights,
        start_candidates,
    ) = candidates_at(0)
    if start_layer is None:
        start_supports = start_heights[start_candidates] + path_height_offset
        selected_start_layer = int(
            start_candidates[np.argmin(np.abs(start_supports - start[2]))]
        )
    else:
        selected_start_layer = int(start_layer)
        if selected_start_layer not in start_candidates:
            raise StairCenterlineError(
                'Requested start layer L{} is not traversable at ({:.3f}, {:.3f})'.format(
                    selected_start_layer, start[0], start[1]
                )
            )

    raw_start_body_z = float(
        start_heights[selected_start_layer] + path_height_offset
    )
    z_alignment = float(start[2] - raw_start_body_z)
    direction = 1 if vertical_span > 0.0 else -1

    selected_layers = []
    selected_costs = []
    aligned_support_z = []
    previous_layer = None
    previous_support_z = None

    for sample_index in range(sample_count):
        x_index, y_index, cell_costs, cell_heights, candidates = candidates_at(
            sample_index
        )

        if sample_index == 0:
            allowed = np.asarray([selected_start_layer], dtype=np.int64)
        else:
            allowed = candidates[
                np.abs(candidates - previous_layer) <= max_layer_step
            ]
            if direction > 0:
                allowed = allowed[allowed >= previous_layer]
            else:
                allowed = allowed[allowed <= previous_layer]

        if sample_index == sample_count - 1 and end_layer is not None:
            requested_end_layer = int(end_layer)
            allowed = allowed[allowed == requested_end_layer]

        if allowed.size == 0:
            candidate_text = ','.join(str(int(value)) for value in candidates[:12])
            if candidates.size > 12:
                candidate_text += ',...'
            raise StairCenterlineError(
                'PCT layer continuity failed at sample {}/{} ({:.3f}, {:.3f}); '
                'previous=L{}, candidates=[{}], max layer step={}'.format(
                    sample_index + 1,
                    sample_count,
                    xy[sample_index, 0],
                    xy[sample_index, 1],
                    previous_layer,
                    candidate_text,
                    max_layer_step,
                )
            )

        supports = (
            cell_heights[allowed] + path_height_offset + z_alignment
        )
        height_errors = np.abs(supports - expected_body_z[sample_index])
        layer_changes = (
            np.zeros_like(allowed, dtype=np.float64)
            if previous_layer is None
            else np.abs(allowed - previous_layer).astype(np.float64)
        )
        order = np.lexsort(
            (
                allowed.astype(np.float64),
                cell_costs[allowed],
                layer_changes,
                height_errors,
            )
        )
        selected_layer = int(allowed[int(order[0])])
        selected_support_z = float(
            cell_heights[selected_layer] + path_height_offset + z_alignment
        )

        if previous_support_z is not None:
            support_step = abs(selected_support_z - previous_support_z)
            if support_step > max_support_step:
                raise StairCenterlineError(
                    'PCT support height jumps {:.3f}m between samples {} and {} '
                    'near ({:.3f}, {:.3f}); limit {:.3f}m'.format(
                        support_step,
                        sample_index,
                        sample_index + 1,
                        xy[sample_index, 0],
                        xy[sample_index, 1],
                        max_support_step,
                    )
                )

        selected_layers.append(selected_layer)
        selected_costs.append(float(cell_costs[selected_layer]))
        aligned_support_z.append(selected_support_z)
        previous_layer = selected_layer
        previous_support_z = selected_support_z

    support_body_z = np.asarray(aligned_support_z, dtype=np.float64)
    window = _odd_window(smoothing_window, sample_count)
    padding = window // 2
    smooth_support = np.convolve(
        np.pad(support_body_z, (padding, padding), mode='edge'),
        np.ones(window, dtype=np.float64) / window,
        mode='valid',
    )
    if direction > 0:
        monotonic_support = np.maximum.accumulate(smooth_support)
    else:
        monotonic_support = np.minimum.accumulate(smooth_support)

    support_span = float(monotonic_support[-1] - monotonic_support[0])
    if abs(support_span) <= 1e-9:
        body_z = expected_body_z.copy()
    else:
        normalized_height = (
            monotonic_support - monotonic_support[0]
        ) / support_span
        body_z = start[2] + normalized_height * vertical_span
    body_z[0] = start[2]
    body_z[-1] = goal[2]

    support_error = np.abs(body_z - support_body_z)
    worst_index = int(np.argmax(support_error))
    worst_error = float(support_error[worst_index])
    if worst_error > max_support_error:
        raise StairCenterlineError(
            'Centerline support error {:.3f}m exceeds {:.3f}m at sample {}/{} '
            '({:.3f}, {:.3f}); path z {:.3f}, support z {:.3f}, layer L{}'.format(
                worst_error,
                max_support_error,
                worst_index + 1,
                sample_count,
                xy[worst_index, 0],
                xy[worst_index, 1],
                body_z[worst_index],
                support_body_z[worst_index],
                selected_layers[worst_index],
            )
        )

    trajectory = np.column_stack((xy, body_z))
    trajectory[0, :3] = start
    trajectory[-1, :3] = goal
    selected_costs = np.asarray(selected_costs, dtype=np.float64)
    selected_layers = np.asarray(selected_layers, dtype=np.int32)
    support_steps = np.abs(np.diff(support_body_z))

    return StairCenterlineResult(
        trajectory=trajectory,
        layers=selected_layers,
        costs=selected_costs,
        support_body_z=support_body_z,
        length=length,
        z_alignment=z_alignment,
        max_cost=float(np.max(selected_costs)),
        max_support_error=worst_error,
        max_support_step=(
            float(np.max(support_steps)) if support_steps.size else 0.0
        ),
    )
