#!/bin/bash
set -euo pipefail

git config --global --add safe.directory '*'
clear

# -------- defaults you can override via env/flags ----------
CURRENT_DIR=$(pwd)
INSTALL_PATH="${INSTALL_PATH:-$HOME/.passes}"

NUM_JOBS=1
BUILD_DOCS=OFF
BUILD_TESTS=ON
BUILD_TOOLS=ON
BUILD_AI=ON
BUILD_TYPE="Release"

LLVM_PREFIX_DEFAULT="$HOME/.local/llvm16"
LLVM_DIR="${LLVM_DIR:-${LLVM_PREFIX_DEFAULT}/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-${LLVM_PREFIX_DEFAULT}/lib/cmake/mlir}"
CLANG_DIR="${CLANG_DIR:-${LLVM_PREFIX_DEFAULT}/lib/cmake/clang}"
INSTALL_DIR="${INSTALL_PATH:-$HOME/.passes}"

OPENBLAS_LIB="${OPENBLAS_LIB:-$HOME/.local/lib/libopenblas.so}"
OPENBLAS_INC="${OPENBLAS_INC:-$HOME/.local/include}"

# -------- CLI args (same switches you had) ----------
while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--jobs) NUM_JOBS="$2"; shift 2 ;;
    --debug)   BUILD_TYPE="Debug"; shift ;;
    --mlir-dir) MLIR_DIR="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --clang-dir) CLANG_DIR="$2"; shift 2 ;;
    --llvm-dir) LLVM_DIR="$2"; shift 2 ;;
    --build-tools) BUILD_TOOLS=ON; shift ;;
    --build-docs)  BUILD_DOCS=ON;  shift ;;
    --build-tests) BUILD_TESTS=ON; shift ;;
    --build-ai)    BUILD_AI=ON;    shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export PATH="$HOME/.local/bin:$PATH"

# -------- AI externals (same logic as yours) ----------
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

# -------- CUDA-Q fetch & configure ----------
BUILD_DIR="${CURRENT_DIR}/build"
DEPS_DIR="${BUILD_DIR}/_deps"
CUDAQ_DIR="${DEPS_DIR}/cuda-quantum"
CUDAQ_REPO="https://github.com/NVIDIA/cuda-quantum.git"

mkdir -p "${BUILD_DIR}" "${DEPS_DIR}"

if [ -d "${CUDAQ_DIR}" ]; then
  echo "[CUDAQ] CUDA Quantum already present at ${CUDAQ_DIR}"
else
  echo "[CUDAQ] Cloning..."
  git clone "${CUDAQ_REPO}" "${CUDAQ_DIR}"
fi

# ensure our user libs are discoverable (zlib/OpenBLAS etc.)
export CMAKE_PREFIX_PATH="$HOME/.local:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
export ZLIB_ROOT="$HOME/.local"
export ZLIB_LIBRARY="$HOME/.local/lib/libz.so"         # or libz.a
export ZLIB_INCLUDE_DIR="$HOME/.local/include"
export LLVM_EXTERNAL_LIT=""                             # neuter lit discovery

FILECHECK="$HOME/.local/llvm16/bin/FileCheck"
LITBIN="$HOME/.local/llvm16/bin/llvm-lit"

# start fresh configure each run to avoid cached test/targets
mkdir -p "${CUDAQ_DIR}/build"
cd "${CUDAQ_DIR}/build"

echo "[CUDAQ] Configuring with Ninja"
CMAKE_ARGS=(
  -G Ninja
  -DMLIR_DIR="${MLIR_DIR}"
  -DClang_DIR="${CLANG_DIR}"
  -DLLVM_DIR="${LLVM_DIR}"

  -DBUILD_TESTING=OFF
  -DLLVM_BUILD_TESTING=OFF
  -DLLVM_INCLUDE_TESTS=OFF
  -DMLIR_INCLUDE_TESTS=OFF
  -DClang_INCLUDE_TESTS=OFF
  -DCMAKE_DISABLE_FIND_PACKAGE_Lit=ON
  -DLLVM_FILECHECK_EXE="${FILECHECK}"

  -DBLA_VENDOR=OpenBLAS
  -DBLAS_LIBRARIES="${OPENBLAS_LIB}"
  -DBLAS_INCLUDE_DIR="${OPENBLAS_INC}"

  # 🔻 turn off remote/HTTP so RestClient never gets used
  -DCUDA_QUANTUM_ENABLE_REMOTE=OFF
  -DCUDAQ_ENABLE_REMOTE=OFF
  -DCUDA_QUANTUM_ENABLE_HTTP_CLIENT=OFF

  # relax warnings that previously tripped -Werror
  -DCMAKE_CXX_FLAGS="-Wno-error=unused-but-set-variable -Wno-unused-but-set-variable"
  -DCMAKE_C_FLAGS="-Wno-error=unused-but-set-variable -Wno-unused-but-set-variable"

  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)


# optional: pass llvm-lit path (harmless if ignored by this CUDA-Q rev)
[ -x "$LITBIN" ] && CMAKE_ARGS+=(-DLLVM_LIT="${LITBIN}")

cmake "${CMAKE_ARGS[@]}" ..

echo "[CUDAQ] Building cudaq-mlir-runtime with ${NUM_JOBS} jobs"
ninja -j"${NUM_JOBS}" cudaq-mlir-runtime

# -------- configure & build your repo ----------
echo "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[MQSS] Configuring CMake"
cmake .. \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
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
