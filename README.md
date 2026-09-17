# ROS 2 Navigation PCT Planner GPL Derivative

This repository publishes the complete source for the modified PCT global
planner used by the LimX ROS 2 navigation integration. It is derived from
[PCT_planner](https://github.com/byangw/PCT_planner) commit
`35cd73fd82bcd51bc538429294af7646b2a09815` and is distributed under GNU GPL
version 2 or, at your option, any later version.

The repository contains the ROS 2 Humble/aarch64 planner, Python wrapper,
pybind11 extensions, tomography tools, build scripts, regression tests, and the
source and notices for retained GTSAM, OSQP/QDLDL, pybind11, and Eigen
snapshots. Read [NOTICE](NOTICE) and [PROVENANCE.md](PROVENANCE.md) before
redistributing it.

## Source snapshot

This repository contains the complete PCT derivative source expected by the
companion LimX ROS 2 navigation integration. Two retained `.gitignore`
exceptions ensure that the offline OSQP/QDLDL configuration headers remain
tracked as corresponding source.

## Build on Ubuntu 22.04 aarch64

Prerequisites include ROS 2 Humble, Python 3.10, CMake, a C++ compiler, NumPy,
SciPy, Open3D, transforms3d, and CUDA/CuPy when generating tomograms on the NX.
Build the retained third-party libraries and planner extensions from the
repository root:

```bash
./planner/build_thirdparty.sh
BUILD_JOBS=2 ./planner/build.sh
```

The build scripts write generated output only under ignored build/install paths
or as ignored extension modules. Run the included tests before deployment:

```bash
python3 -m unittest discover -s planner/tests -p 'test_*.py'
python3 -m unittest discover -s tomography/tests -p 'test_*.py'
```

## ROS 2 runtime

Provide a separately generated and validated tomogram; maps are runtime data and
are intentionally not distributed here. Then source ROS 2 Humble and start the
planner:

```bash
export PCT_TOMOGRAM=/absolute/path/to/map.pickle
./planner/run_navigation_humble.sh
```

The navigation wrapper consumes standard pose/goal messages and publishes
`nav_msgs/Path` on `/pct_path`. The permissive integration repository and the
FAST-LIO GPL repository are checked out and built separately. This process-level
engineering separation does not change any component's license.

Companion repositories:

- <https://github.com/limxdynamics/tron2-navigation-ros2>
- <https://github.com/1392559136a-jpg/tron2-navigation-fastlio-gpl>

## Dependency provenance

[PROVENANCE.md](PROVENANCE.md) records file counts, exact baseline hashes, and
local differences for all retained third-party source. To reproduce the
byte-level comparison, provide a separate upstream PCT clone containing the
pinned commit:

```bash
./tools/verify_pct_dependency_provenance.sh /path/to/upstream/PCT_planner
```

The script performs no network access.

## Source integrity and license

`SOURCE_INVENTORY.sha256` locks every distributed file except itself.
`./audit_source.sh` verifies the inventory, required GPL and third-party license
texts, source-only contents, privacy exclusions, and path constraints.

PCT is GPL-2.0-or-later. Embedded dependencies remain under their own terms,
including BSD-3-Clause, Apache-2.0, BSD-style, MPL-2.0, and retained per-file
BSD/LGPL notices. The software is provided without warranty.
