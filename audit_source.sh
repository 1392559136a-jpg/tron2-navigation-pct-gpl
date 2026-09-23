#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail() { printf 'GPL_SOURCE_AUDIT_FAIL: %s\n' "$*" >&2; exit 1; }

for command_name in file git sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || fail "missing command: ${command_name}"
done

required=(
  LICENSE NOTICE PROVENANCE.md README.md SOURCE_INVENTORY.sha256
  planner/build.sh planner/build_thirdparty.sh
  planner/run_navigation_humble.sh planner/scripts/navigation_plan.py
  planner/scripts/planner_wrapper.py planner/scripts/stair_centerline.py
  planner/tests/test_stair_centerline.py planner/config/navigation.yaml
  planner/lib/src/a_star/a_star_search.cc planner/lib/src/a_star/a_star_search.h
  planner/lib/3rdparty/gtsam-4.1.1/LICENSE.BSD
  planner/lib/3rdparty/osqp/LICENSE planner/lib/3rdparty/osqp/NOTICE
  planner/lib/3rdparty/osqp/include/osqp_configure.h
  planner/lib/3rdparty/osqp/lin_sys/direct/qdldl/qdldl_sources/include/qdldl_types.h
  planner/lib/3rdparty/pybind11/LICENSE
  planner/lib/3rdparty/gtsam-4.1.1/gtsam/3rdparty/Eigen/COPYING.README
  planner/lib/3rdparty/gtsam-4.1.1/gtsam/3rdparty/Eigen/COPYING.MPL2
  planner/lib/3rdparty/gtsam-4.1.1/gtsam/3rdparty/Eigen/COPYING.BSD
  planner/lib/3rdparty/gtsam-4.1.1/gtsam/3rdparty/Eigen/COPYING.LGPL
  tools/verify_pct_dependency_provenance.sh
)
for relative in "${required[@]}"; do
  [[ -f "${ROOT}/${relative}" ]] || fail "missing required source or license: ${relative}"
done

grep -Fq 'GNU GENERAL PUBLIC LICENSE' "${ROOT}/LICENSE" || fail 'GPLv2 text missing'
grep -Fq 'either version 2 of the License, or' "${ROOT}/NOTICE" \
  || fail 'GPL-2.0-or-later notice missing'
grep -Fq '35cd73fd82bcd51bc538429294af7646b2a09815' \
  "${ROOT}/PROVENANCE.md" || fail 'PCT baseline commit missing'
for marker in \
  'gtsam 4.1.1' 'osqp 0.6.2' 'pybind11 2.11.0.dev1' \
  'Eigen embedded by GTSAM'; do
  grep -Fq "${marker}" "${ROOT}/NOTICE" || fail "dependency notice missing: ${marker}"
done

(
  cd "${ROOT}"
  sha256sum -c SOURCE_INVENTORY.sha256 >/dev/null
) || fail 'source inventory mismatch'
inventory_paths="$(mktemp)"
actual_paths="$(mktemp)"
trap 'rm -f "${inventory_paths}" "${actual_paths}"' EXIT
awk '{sub(/^\.\//, "", $2); print $2}' "${ROOT}/SOURCE_INVENTORY.sha256" \
  | LC_ALL=C sort >"${inventory_paths}"
find "${ROOT}" -type f ! -path '*/.git/*' ! -name SOURCE_INVENTORY.sha256 \
  -printf '%P\n' | LC_ALL=C sort >"${actual_paths}"
cmp -s "${inventory_paths}" "${actual_paths}" || fail 'inventory file set mismatch'

if find "${ROOT}" -type d ! -path '*/.git/*' \
  \( -name build -o -name install -o -name obj -o -name __pycache__ \
     -o -name .pytest_cache \) -print -quit | grep -q .; then
  fail 'generated directory found'
fi
if find "${ROOT}" -type f ! -path '*/.git/*' \
  \( -name '*.o' -o -name '*.obj' -o -name '*.a' -o -name '*.so' \
     -o -name '*.pyc' -o -name '*.pcd' -o -name '*.pickle' -o -name '*.pkl' \
     -o -name '*.bag' -o -name '*.db3' -o -name '*.mcap' \) \
  -print -quit | grep -q .; then
  fail 'generated binary, map, or recording found'
fi
if find "${ROOT}" -type f ! -path '*/.git/*' -size +25M -print -quit | grep -q .; then
  fail 'file larger than 25 MiB found'
fi
while IFS= read -r relative; do
  (( ${#relative} <= 180 )) || fail "path exceeds 180 characters: ${relative}"
done < <(find "${ROOT}" -mindepth 1 ! -path '*/.git/*' -printf '%P\n')

private_value_pattern='(^|[^0-9])(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3})([^0-9]|$)|WF_TRON[A-Za-z0-9_]*|guest@|/var/limx'
machine_path_pattern='/home/[A-Za-z0-9_.-]+(/|[^A-Za-z0-9_])'
sensitive_output="$(
  {
    grep -RInI -E --exclude=audit_source.sh --exclude=SOURCE_INVENTORY.sha256 \
      --exclude-dir=.git -- "${private_value_pattern}" "${ROOT}" || true
    grep -RInI -E --exclude=audit_source.sh --exclude=SOURCE_INVENTORY.sha256 \
      --exclude-dir=.git --exclude-dir=3rdparty \
      -- "${machine_path_pattern}" "${ROOT}" || true
  } | grep -v -F 'GeneratedCodeAttribute("Microsoft.VisualStudio.Editors.SettingsDesigner.SettingsSingleFileGenerator", "10.0.0.0")' \
    || true
)"
if [[ -n "${sensitive_output}" ]]; then
  printf '%s\n' "${sensitive_output}" >&2
  fail 'private address, identity, or machine path found'
fi

echo 'GPL_SOURCE_AUDIT=PASS unit=PCT_PLANNER'
