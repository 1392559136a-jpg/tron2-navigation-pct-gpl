#!/bin/bash

# 开启错误检测，遇到编译错误立即退出
set -e

# 获取脚本根目录（确保路径无空格、解析正确）
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "ROOT_DIR: ${ROOT_DIR}"

# ========== 可选：临时创建2GB swap（内存不足时启用） ==========
# echo "Creating temporary swap file for compilation..."
# sudo fallocate -l 2G /tmp/gtsam_temp_swap
# sudo chmod 600 /tmp/gtsam_temp_swap
# sudo mkswap /tmp/gtsam_temp_swap
# sudo swapon /tmp/gtsam_temp_swap

# ========== 编译gtsam-4.1.1 ==========
echo "Start building gtsam-4.1.1..."
GTSAM_DIR="${ROOT_DIR}/lib/3rdparty/gtsam-4.1.1"
# 先删除旧目录，避免残留问题
rm -rf "${GTSAM_DIR}/build" "${GTSAM_DIR}/install"
# 确保目录创建正确（无空格）
mkdir -p "${GTSAM_DIR}/build" "${GTSAM_DIR}/install"
cd "${GTSAM_DIR}/build"

# 关键修正：cmake参数行格式（反斜杠后无空格，参数连续）
cmake .. \
  -DCMAKE_INSTALL_PREFIX="../install" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O2 -DEIGEN_NO_DEBUG -DNDEBUG" \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_BUILD_EXAMPLES=OFF \
  -DGTSAM_BUILD_TESTS=OFF

# 降低并行数，避免OOM
make -j2 && make install
echo "gtsam-4.1.1 build completed!"

# ========== 编译osqp ==========
echo "Start building osqp..."
OSQP_DIR="${ROOT_DIR}/lib/3rdparty/osqp"
rm -rf "${OSQP_DIR}/build" "${OSQP_DIR}/install"
mkdir -p "${OSQP_DIR}/build"
cd "${OSQP_DIR}/build"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="../install" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O2"

make -j2 && make install
echo "osqp build completed!"

# ========== 可选：清理临时swap ==========
# echo "Cleaning temporary swap file..."
# sudo swapoff /tmp/gtsam_temp_swap
# sudo rm -f /tmp/gtsam_temp_swap

echo "All dependencies (gtsam + osqp) built successfully!"

