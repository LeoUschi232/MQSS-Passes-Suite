#!/bin/bash
set -euo pipefail

git config --global --add safe.directory '*'
clear

# ======================= Defaults =======================
CURRENT_DIR="$(pwd)"
INSTALL_PATH="${INSTALL_PATH:-$HOME/.passes}"

NUM_JOBS="${NUM_JOBS:-1}"
BUILD_DOCS="${BUILD_DOCS:-OFF}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
BUILD_TOOLS="${BUILD_TOOLS:-ON}"
BUILD_AI="${BUILD_AI:-ON}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Point to your user LLVM16 install (adjust if you put it elsewhere)
LLVM_PREFIX_DEFAULT="$HOME/.local/llvm16"
LLVM_DIR="${LLVM_DIR:-${LLVM_PREFIX_DEFAULT}/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-${LLVM_PREFIX_DEFAULT}/lib/cmake/mlir}"
CLANG_DIR="${CLANG_DIR:-${LLVM_PREFIX_DEFAULT}/lib/cmake/clang}"
INSTALL_DIR="${INSTALL_PATH}"

# Optional: user OpenBLAS (you built it under ~/.local/lib)
OPENBLAS_LIB="${OPENBLAS_LIB:-$HOME/.local/lib/libopenblas.so}"
OPENBLAS_INC="${OPENBLAS_INC:-$HOME/.local/include}"

# User zlib (you built 1.3.1 under ~/.local)
ZLIB_ROOT_DIR="${ZLIB_ROOT_DIR:-$HOME/.local}"
ZLIB_LIB="${ZLIB_LIB:-$HOME/.local/lib/libz.so}"   # or libz.a
ZLIB_INC="${ZLIB_INC:-$HOME/.local/include}"

# Make user cmake/ninja visible (installed via pip --user)
export PATH="$HOME/.local/bin:$PATH"

# =================== AI externals =======================
AI_DIR="${CURRENT_DIR}/AI"
AI_EXTERNAL_DIR="${AI_DIR}/external"
LIBTORCH_DIR="${AI_EXTERNAL_DIR}/libtorch"
TENSORFLOW_DIR="${AI_EXTERNAL_DIR}/tensorflow"
mkdir -p "${AI_EXTERNAL_DIR}"

if [ ! -d "${LIBTORCH_DIR}" ]; then
  cd "${AI_EXTERNAL_DIR}"
  echo "[AI] Downloading libtorch..."
  wget -q https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip -q libtorch-shared-with-deps-latest.zip
  rm -f libtorch-shared-with-deps-latest.zip
else
  echo "[AI] Libtorch already present at ${LIBTORCH_DIR}"
fi

if [ ! -d "${TENSORFLOW_DIR}" ]; then
  cd "${AI_EXTERNAL_DIR}"
  echo "[AI] Building tensorflow-cpp (user space)..."
  git clone https://github.com/leggedrobotics/tensorflow-cpp.git
  cd tensorflow-cpp/eigen
  ./install.sh --run-cmake
  cd ../tensorflow
  mkdir -p build && cd build
  cmake -DCMAKE_INSTALL_PREFIX="${TENSORFLOW_DIR}" -DCMAKE_BUILD_TYPE=Release ..
  make install -j"$(nproc)"
  cd "${AI_EXTERNAL_DIR}"
  rm -rf tensorflow-cpp
else
  echo "[AI] Tensorflow already present at ${TENSORFLOW_DIR}"
fi

cd "${CURRENT_DIR}"

# =================== CUDA-Q fetch =======================
BUILD_DIR="${CURRENT_DIR}/build"
DEPS_DIR="${BUILD_DIR}/_deps"
CUDAQ_DIR="${DEPS_DIR}/cuda-quantum"
CUDAQ_REPO="https://github.com/NVIDIA/cuda-quantum.git"

mkdir -p "${BUILD_DIR}" "${DEPS_DIR}"

if [ ! -d "${CUDAQ_DIR}" ]; then
  echo "[CUDAQ] Cloning CUDA Quantum into ${CUDAQ_DIR}"
  git clone "${CUDAQ_REPO}" "${CUDAQ_DIR}"
else
  echo "[CUDAQ] CUDA Quantum already present at ${CUDAQ_DIR}"
fi

# Clean CUDA-Q build dir to avoid cached bad state
rm -rf "${CUDAQ_DIR}/build"
mkdir -p "${CUDAQ_DIR}/build"
cd "${CUDAQ_DIR}/build"

echo "[CUDAQ] Configuring with Ninja"

# ---------- Make user libs discoverable ----------
export CMAKE_PREFIX_PATH="${ZLIB_ROOT_DIR}:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="${ZLIB_ROOT_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${ZLIB_ROOT_DIR}/lib:${LD_LIBRARY_PATH:-}"

# zlib hints
export ZLIB_ROOT="${ZLIB_ROOT_DIR}"
export ZLIB_LIBRARY="${ZLIB_LIB}"
export ZLIB_INCLUDE_DIR="${ZLIB_INC}"

# lit/FileCheck handling (we're using prebuilt LLVM binaries)
export LLVM_EXTERNAL_LIT=""                                  # neuter lit discovery
export LLVM_FILECHECK_EXE="${LLVM_FILECHECK_EXE:-$HOME/.local/llvm16/bin/FileCheck}"

# Create a tiny CMake shim that defines a dummy FileCheck target and disables lit
OVR_DIR="$HOME/.cmake-overrides"
OVR_FILE="$OVR_DIR/force_no_lit.cmake"
mkdir -p "$OVR_DIR"
cat > "$OVR_FILE" <<'EOF'
# Provide a dummy CMake target so "add_dependencies(... FileCheck)" doesn't fail.
if(NOT TARGET FileCheck)
  add_custom_target(FileCheck)
endif()

# Expose a path to the FileCheck executable if available from env.
if(NOT LLVM_FILECHECK_EXE AND DEFINED ENV{LLVM_FILECHECK_EXE})
  set(LLVM_FILECHECK_EXE "$ENV{LLVM_FILECHECK_EXE}" CACHE FILEPATH "Path to FileCheck" FORCE)
endif()
if(NOT FILECHECK_EXE AND DEFINED ENV{LLVM_FILECHECK_EXE})
  set(FILECHECK_EXE "$ENV{LLVM_FILECHECK_EXE}" CACHE FILEPATH "Path to FileCheck" FORCE)
endif()

# Neuter all lit/testing hooks unconditionally.
set(LLVM_EXTERNAL_LIT "" CACHE FILEPATH "Disable lit" FORCE)
set(LLVM_LIT "" CACHE FILEPATH "Disable lit" FORCE)
set(LIT_EXECUTABLE "" CACHE FILEPATH "Disable lit" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
EOF

# --------------- Configure CUDA-Q ----------------------
cmake -G Ninja \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DClang_DIR="${CLANG_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  \
  -DBUILD_TESTING=OFF \
  -DLLVM_BUILD_TESTING=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DClang_INCLUDE_TESTS=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_Lit=ON \
  -DLLVM_FILECHECK_EXE="${LLVM_FILECHECK_EXE}" \
  -DFILECHECK_EXE="${LLVM_FILECHECK_EXE}" \
  \
  -DBLA_VENDOR=OpenBLAS \
  -DBLAS_LIBRARIES="${OPENBLAS_LIB}" \
  -DBLAS_INCLUDE_DIR="${OPENBLAS_INC}" \
  \
  -DZLIB_ROOT="${ZLIB_ROOT_DIR}" \
  -DZLIB_LIBRARY="${ZLIB_LIB}" \
  -DZLIB_INCLUDE_DIR="${ZLIB_INC}" \
  \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="${OVR_FILE}" \
  ..

echo "[CUDAQ] Building cudaq-mlir-runtime with ${NUM_JOBS} jobs"
ninja -j"${NUM_JOBS}" cudaq-mlir-runtime

# ============ Configure & build your repo =============
echo "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[MQSS] Configuring CMake"
cmake .. \
  -DCMAKE_C_COMPILER="${CC:-gcc}" \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DBUILD_MLIR_PASSES_TOOLS="${BUILD_TOOLS}" \
  -DBUILD_MLIR_PASSES_DOCS="${BUILD_DOCS}" \
  -DBUILD_MLIR_PASSES_TESTS="${BUILD_TESTS}" \
  -DBUILD_MLIR_PASSES_AI="${BUILD_AI}" \
  -DCUDAQ_SOURCE_DIR="${CUDAQ_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "[MQSS] Building with ${NUM_JOBS} jobs"
make -j"${NUM_JOBS}"

echo "[DONE] Build completed successfully."
