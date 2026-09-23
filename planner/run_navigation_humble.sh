#!/usr/bin/env bash
set -euo pipefail

PLANNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
DEFAULT_TOMOGRAM="${PLANNER_ROOT}/../rsc/tomogram/map.pickle"
LEGACY_TOMOGRAM="${PLANNER_ROOT}/../../PCT_planner/rsc/tomogram/map.pickle"
if [[ -n "${PCT_TOMOGRAM:-}" ]]; then
  TOMOGRAM="${PCT_TOMOGRAM}"
elif [[ -f "${DEFAULT_TOMOGRAM}" ]]; then
  TOMOGRAM="${DEFAULT_TOMOGRAM}"
else
  TOMOGRAM="${LEGACY_TOMOGRAM}"
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS 2 Humble setup not found: ${ROS_SETUP}" >&2
  exit 1
fi
if [[ ! -f "${TOMOGRAM}" ]]; then
  echo "PCT tomogram not found: ${TOMOGRAM}" >&2
  echo "Copy the validated map.pickle into rsc/tomogram or set PCT_TOMOGRAM." >&2
  exit 1
fi

set +u
source "${ROS_SETUP}"
set -u
export PYTHONPATH="${PLANNER_ROOT}:${PLANNER_ROOT}/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PLANNER_ROOT}/lib/3rdparty/gtsam-4.1.1/install/lib:${PLANNER_ROOT}/lib/3rdparty/osqp/install/lib:${PLANNER_ROOT}/lib/build/src/common/smoothing:${LD_LIBRARY_PATH:-}"

exec python3 "${PLANNER_ROOT}/scripts/navigation_plan.py" \
  --tomogram "${TOMOGRAM}" \
  --ros-args --params-file "${PLANNER_ROOT}/config/navigation.yaml" "$@"