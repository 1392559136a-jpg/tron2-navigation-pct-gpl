#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "ROOT_DIR: ${ROOT_DIR}"

cd "${ROOT_DIR}/lib"

rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build build --parallel "${BUILD_JOBS:-2}"
rm -f a_star*.so traj_opt*.so ele_planner*.so py_map_manager*.so libcommon_smoothing.so
cp build/src/a_star/a_star*.so .
cp build/src/trajectory_optimization/traj_opt*.so .
cp build/src/ele_planner/ele_planner*.so .
cp build/src/map_manager/py_map_manager*.so .
cp build/src/common/smoothing/libcommon_smoothing.so .

# # optional
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROOT_DIR}/lib/3rdparty/gtsam-4.1.1/install/lib
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ROOT_DIR}/lib/build/src/common/smoothing"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/lib"
# pybind11-stubgen -o ./ a_star
# pybind11-stubgen -o ./ traj_opt
# pybind11-stubgen -o ./ ele_planner
# pybind11-stubgen -o ./ py_map_manager
# cp ./a_star-stubs/__init__.pyi ./a_star.pyi
# cp ./traj_opt-stubs/__init__.pyi ./traj_opt.pyi
# cp ./ele_planner-stubs/__init__.pyi ./ele_planner.pyi
# cp ./py_map_manager-stubs/__init__.pyi ./py_map_manager.pyi
