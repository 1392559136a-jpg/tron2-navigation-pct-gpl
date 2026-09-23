#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_THIRDPARTY="${ROOT_DIR}/planner/lib/3rdparty"
BASELINE_COMMIT="35cd73fd82bcd51bc538429294af7646b2a09815"
UPSTREAM_REPOSITORY="${1:-${ROOT_DIR}/../PCT_planner}"

usage() {
  cat <<EOF
Usage: ./tools/verify_pct_dependency_provenance.sh [UPSTREAM_PCT_CLONE]

UPSTREAM_PCT_CLONE must be a Git clone containing PCT_planner commit
${BASELINE_COMMIT}. The script performs no network access and compares the
public-source dependency subset byte-for-byte with that recorded baseline.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
[[ $# -le 1 ]] || { usage >&2; exit 2; }
[[ -d "${LOCAL_THIRDPARTY}" ]] || {
  echo "Local PCT third-party tree not found: ${LOCAL_THIRDPARTY}" >&2
  exit 2
}
git -C "${UPSTREAM_REPOSITORY}" cat-file -e "${BASELINE_COMMIT}^{commit}" \
  2>/dev/null || {
    echo "The upstream clone does not contain ${BASELINE_COMMIT}: ${UPSTREAM_REPOSITORY}" >&2
    exit 2
  }
for command in git rsync sha256sum tar; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "Required command not found: ${command}" >&2
    exit 2
  }
done

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT INT TERM
mkdir -p "${TEMP_DIR}/raw"
git -C "${UPSTREAM_REPOSITORY}" archive "${BASELINE_COMMIT}" \
  planner/lib/3rdparty | tar -xf - -C "${TEMP_DIR}/raw"
UPSTREAM_THIRDPARTY="${TEMP_DIR}/raw/planner/lib/3rdparty"

# Keep this aligned with create_public_source_bundle.sh. README files are
# excluded only from this source comparison because the public exporter rewrites
# their omitted-media links to the pinned upstream snapshots.
comparison_excludes=(
  --exclude='.git'
  --exclude='.git/'
  --exclude='.svn/'
  --exclude='.github/'
  --exclude='.idea/'
  --exclude='.vscode/'
  --exclude='.cache/'
  --exclude='.pytest_cache/'
  --exclude='__pycache__/'
  --exclude='build/'
  --exclude='install/'
  --exclude='devel/'
  --exclude='log/'
  --exclude='Log/'
  --exclude='doc/'
  --exclude='PCD/'
  --exclude='maps/'
  --exclude='bag/'
  --exclude='pcd/'
  --exclude='tomogram/'
  --exclude='handoff/'
  --exclude='.build/'
  --exclude='obj/'
  --exclude='navigation.env'
  --exclude='README*'
  --exclude='*.o'
  --exclude='*.obj'
  --exclude='*.a'
  --exclude='*.so'
  --exclude='*.pyc'
  --exclude='*.pcd'
  --exclude='*.ply'
  --exclude='*.stl'
  --exclude='*.pickle'
  --exclude='*.pkl'
  --exclude='*.npy'
  --exclude='*.bag'
  --exclude='*.db3'
  --exclude='*.mcap'
  --exclude='*.lvx'
  --exclude='*.gif'
  --exclude='*.zip'
  --exclude='*.tar'
  --exclude='*.tar.gz'
  --exclude='*.tgz'
  --exclude='*.pdf'
  --exclude='*.docx'
  --exclude='*.xls'
  --exclude='*.xlsx'
  --exclude='rs_driverConfig.cmake'
  --exclude='rs_driverConfigVersion.cmake'
)

assert_equal() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'PROVENANCE_MISMATCH: %s expected=%s actual=%s\n' \
      "${label}" "${expected}" "${actual}" >&2
    exit 1
  fi
}

assert_array_equal() {
  local label="$1"
  local expected_name="$2"
  local actual_name="$3"
  local -n expected_ref="${expected_name}"
  local -n actual_ref="${actual_name}"
  assert_equal "${label}.count" "${#actual_ref[@]}" "${#expected_ref[@]}"
  local index
  for index in "${!expected_ref[@]}"; do
    assert_equal "${label}[${index}]" \
      "${actual_ref[${index}]}" "${expected_ref[${index}]}"
  done
}

compare_dependency() {
  local dependency="$1"
  local expected_upstream_count="$2"
  local expected_local_count="$3"
  local expected_upstream_inventory="$4"
  local expected_local_inventory="$5"
  local expected_added_name="$6"
  local expected_changed_name="$7"
  local expected_deleted_name="$8"
  local upstream_copy="${TEMP_DIR}/upstream-${dependency}"
  local local_copy="${TEMP_DIR}/local-${dependency}"
  local upstream_inventory="${TEMP_DIR}/upstream-${dependency}.sha256"
  local local_inventory="${TEMP_DIR}/local-${dependency}.sha256"
  local upstream_paths="${TEMP_DIR}/upstream-${dependency}.paths"
  local local_paths="${TEMP_DIR}/local-${dependency}.paths"
  local added=()
  local changed=()
  local deleted=()
  local relative_path

  mkdir -p "${upstream_copy}" "${local_copy}"
  rsync -a --prune-empty-dirs "${comparison_excludes[@]}" \
    "${UPSTREAM_THIRDPARTY}/${dependency}/" "${upstream_copy}/"
  rsync -a --prune-empty-dirs "${comparison_excludes[@]}" \
    "${LOCAL_THIRDPARTY}/${dependency}/" "${local_copy}/"

  (
    cd "${upstream_copy}"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) >"${upstream_inventory}"
  (
    cd "${local_copy}"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) >"${local_inventory}"
  (
    cd "${upstream_copy}"
    find . -type f -printf '%P\n' | LC_ALL=C sort
  ) >"${upstream_paths}"
  (
    cd "${local_copy}"
    find . -type f -printf '%P\n' | LC_ALL=C sort
  ) >"${local_paths}"

  mapfile -t added < <(LC_ALL=C comm -13 "${upstream_paths}" "${local_paths}")
  mapfile -t deleted < <(LC_ALL=C comm -23 "${upstream_paths}" "${local_paths}")
  while IFS= read -r relative_path; do
    if ! cmp -s "${upstream_copy}/${relative_path}" \
      "${local_copy}/${relative_path}"; then
      changed+=("${relative_path}")
    fi
  done < <(LC_ALL=C comm -12 "${upstream_paths}" "${local_paths}")

  local upstream_count local_count upstream_hash local_hash
  upstream_count="$(wc -l <"${upstream_inventory}")"
  local_count="$(wc -l <"${local_inventory}")"
  upstream_hash="$(sha256sum "${upstream_inventory}" | cut -d' ' -f1)"
  local_hash="$(sha256sum "${local_inventory}" | cut -d' ' -f1)"

  assert_equal "${dependency}.upstream_files" \
    "${upstream_count}" "${expected_upstream_count}"
  assert_equal "${dependency}.local_files" \
    "${local_count}" "${expected_local_count}"
  assert_equal "${dependency}.upstream_inventory" \
    "${upstream_hash}" "${expected_upstream_inventory}"
  assert_equal "${dependency}.local_inventory" \
    "${local_hash}" "${expected_local_inventory}"
  assert_array_equal "${dependency}.added" "${expected_added_name}" added
  assert_array_equal "${dependency}.changed" "${expected_changed_name}" changed
  assert_array_equal "${dependency}.deleted" "${expected_deleted_name}" deleted

  printf 'DEPENDENCY_PROVENANCE=PASS dependency=%s upstream_files=%s local_files=%s added=%s changed=%s deleted=%s\n' \
    "${dependency}" "${upstream_count}" "${local_count}" \
    "${#added[@]}" "${#changed[@]}" "${#deleted[@]}"
}

none=()
gtsam_added=()
gtsam_changed=(package.xml wrap/setup.py)
gtsam_deleted=()
osqp_added=(
  include/osqp_configure.h
  lin_sys/direct/qdldl/qdldl_sources/include/qdldl_types.h
)
osqp_changed=(
  include/.gitignore
  lin_sys/direct/qdldl/qdldl_sources/include/.gitignore
)
osqp_deleted=()
pybind_added=()
pybind_changed=()
pybind_deleted=()

compare_dependency \
  gtsam-4.1.1 3824 3824 \
  be3ce1999a1407da4d5ee332c8096bbf12a42163f129e94e55f4c9158d114b90 \
  ce3e4cd5782d656ff9ff23cea2fb5f8b4ad8ed4cc10f48ae7265dc2d930603ee \
  gtsam_added gtsam_changed gtsam_deleted
compare_dependency \
  osqp 212 214 \
  6069f33b0505e6ca9c568800b8516901400e4c2a5ab6581a520a766ff6d86cac \
  c39a52f3b64afdd9e9746d0383aac08ad32600df88569e367306debd119f78a2 \
  osqp_added osqp_changed osqp_deleted
compare_dependency \
  pybind11 230 230 \
  5748f6c228cc20461ae70bb2ca7040e9a138c7aa205cd740f5a98db07fa48593 \
  5748f6c228cc20461ae70bb2ca7040e9a138c7aa205cd740f5a98db07fa48593 \
  pybind_added pybind_changed pybind_deleted

# Eigen is embedded below GTSAM. Its retained 1,147 files are byte-identical to
# the pinned PCT baseline. Its COPYING.README says the tree is primarily MPL-2.0
# but also contains BSD/LGPL files, so the complete upstream license set remains.
EIGEN_HASH="da583b0f3959c39b2de8a150707f4e002ffd28f7caa7867e6d574867a9c8154e"
eigen_upstream="${TEMP_DIR}/upstream-gtsam-4.1.1/gtsam/3rdparty/Eigen"
eigen_local="${TEMP_DIR}/local-gtsam-4.1.1/gtsam/3rdparty/Eigen"
(
  cd "${eigen_upstream}"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) >"${TEMP_DIR}/eigen-upstream.sha256"
(
  cd "${eigen_local}"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) >"${TEMP_DIR}/eigen-local.sha256"
assert_equal "eigen.file_count" \
  "$(wc -l <"${TEMP_DIR}/eigen-local.sha256")" 1147
assert_equal "eigen.upstream_inventory" \
  "$(sha256sum "${TEMP_DIR}/eigen-upstream.sha256" | cut -d' ' -f1)" \
  "${EIGEN_HASH}"
assert_equal "eigen.local_inventory" \
  "$(sha256sum "${TEMP_DIR}/eigen-local.sha256" | cut -d' ' -f1)" \
  "${EIGEN_HASH}"

echo "EIGEN_PROVENANCE=PASS files=1147 byte_identical_to_pct_baseline=1"
echo "PCT_DEPENDENCY_PROVENANCE=PASS baseline=${BASELINE_COMMIT}"
