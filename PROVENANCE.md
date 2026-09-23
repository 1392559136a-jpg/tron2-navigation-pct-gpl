# PCT RC2026 Source Provenance

## Baseline

- Upstream: <https://github.com/byangw/PCT_planner.git>
- Baseline commit: `35cd73fd82bcd51bc538429294af7646b2a09815`
- Upstream/local license: GPL version 2 or later as stated in NOTICE; preserve
  the complete GPLv2 text in LICENSE.

The recorded baseline is a provenance reference, not a drop-in replacement for
this ROS 2 Humble/aarch64 runtime derivative.

## Local modification scope (2026)

The `PCT_planner-RC2026_Map_Planner` derivative contains:

- ROS 2 Humble/aarch64 build and launch scripts;
- runtime navigation configuration and path publication;
- C++ planner, map-manager, optimizer, smoothing, and portability changes;
- Python planner wrappers and conversion/visualization changes;
- offline PCD-to-tomogram processing, goal editing, and chunk transport tools;
- ROS 2 RViz configuration; and
- regression tests for the ROS 2 navigation and tomography interfaces.

The dependency licenses and upstream dependency modifications are documented in
NOTICE and in each `planner/lib/3rdparty` dependency directory.

## Verified vendored-dependency comparison

On 2026-09-14, the retained dependency source was compared byte-for-byte with
the Git objects at the pinned PCT commit. The comparison uses `git archive`
rather than an existing checkout's working files, applies the same source-only
filters as the public exporter, and excludes README files only because their
omitted-media links are rewritten during public staging.

Results:

- GTSAM 4.1.1: 3,824 upstream and 3,824 local retained files; no added or
  deleted paths. Only `package.xml` and `wrap/setup.py` differ, and both changes
  replace ambiguous `BSD` metadata with `BSD-3-Clause`.
  - upstream inventory SHA-256:
    `be3ce1999a1407da4d5ee332c8096bbf12a42163f129e94e55f4c9158d114b90`
  - local inventory SHA-256:
    `ce3e4cd5782d656ff9ff23cea2fb5f8b4ad8ed4cc10f48ae7265dc2d930603ee`
- OSQP 0.6.2: 212 upstream and 214 local retained files. The two added
  build-configuration headers are listed in NOTICE. The only modified upstream
  paths are their two parent `.gitignore` files, which unignore those headers so
  a normal `git add .` retains the complete offline-build source; no production
  or dependency code is modified or deleted.
  - upstream inventory SHA-256:
    `6069f33b0505e6ca9c568800b8516901400e4c2a5ab6581a520a766ff6d86cac`
  - local inventory SHA-256:
    `c39a52f3b64afdd9e9746d0383aac08ad32600df88569e367306debd119f78a2`
- pybind11 2.11.0.dev1: all 230 retained files are byte-identical.
  - upstream/local inventory SHA-256:
    `5748f6c228cc20461ae70bb2ca7040e9a138c7aa205cd740f5a98db07fa48593`
- Eigen embedded by GTSAM: all 1,147 retained files are byte-identical.
  - upstream/local inventory SHA-256:
    `da583b0f3959c39b2de8a150707f4e002ffd28f7caa7867e6d574867a9c8154e`

Run the offline [verification script](tools/verify_pct_dependency_provenance.sh)
against any Git clone containing the pinned PCT commit to reproduce all counts,
path differences, and aggregate hashes. The script performs no clone or other
network operation.

Eigen's included `COPYING.README` states that Eigen is primarily MPL-2.0 but
contains some BSD- and LGPL-licensed files. It recommends defining
`EIGEN_MPL2_ONLY` to guarantee an MPL-2.0/permissive-only include set. The
current inherited GTSAM build does not define that option, so this distribution
conservatively preserves and documents the complete upstream `COPYING.*`
license set instead of claiming that all Eigen files are MPL-2.0-only.

## Release contents

A release archive must contain this file, NOTICE, LICENSE, the modified source,
headers, build scripts, and interface files. It must not contain maps,
tomograms, recordings, handoff files, or locally built binaries.
