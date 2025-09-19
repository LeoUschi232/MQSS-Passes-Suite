#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
export PATH="$HOME/.local/bin:$HOME/.local/llvm16/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$HOME/.local/llvm16/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$HOME/.local:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# LLVM/MLIR package dirs
export LLVM_DIR="${LLVM_DIR:-$HOME/.local/llvm16/lib/cmake/llvm}"
export MLIR_DIR="${MLIR_DIR:-$HOME/.local/llvm16/lib/cmake/mlir}"
export CLANG_DIR="${CLANG_DIR:-$HOME/.local/llvm16/lib/cmake/clang}"

# Compilers (use clang toolchain)
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"

# CMake generator
: "${CMAKE_GENERATOR:=Ninja}"

# Jobs / build type
JOBS=8
BUILD_TYPE=Release
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2;;
    --debug)   BUILD_TYPE=Debug; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# -------- Paths --------
REPO_ROOT="$(pwd)"
BUILD_DIR="$REPO_ROOT/build"
DEPS_DIR="$BUILD_DIR/_deps"
CUDAQ_DIR="$DEPS_DIR/cuda-quantum"

# -------- Optional AI externals (libtorch/tensorflow) --------
AI_DIR="$REPO_ROOT/AI"
AI_EXT="$AI_DIR/external"
LIBTORCH_DIR="$AI_EXT/libtorch"
TENSORFLOW_DIR="$AI_EXT/tensorflow"

mkdir -p "$AI_EXT"
if [[ ! -d "$LIBTORCH_DIR" ]]; then
  echo "[AI] Installing LibTorch (CPU, nightly) into $AI_EXT ..."
  pushd "$AI_EXT" >/dev/null
  wget -q https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip -q libtorch-shared-with-deps-latest.zip
  rm -f libtorch-shared-with-deps-latest.zip
  popd >/dev/null
else
  echo "[AI] Libtorch already present at $LIBTORCH_DIR"
fi

if [[ ! -d "$TENSORFLOW_DIR" ]]; then
  echo "[AI] Installing TensorFlow C++ shim into $TENSORFLOW_DIR ..."
  pushd "$AI_EXT" >/dev/null
  git clone https://github.com/leggedrobotics/tensorflow-cpp.git
  pushd tensorflow-cpp/eigen >/dev/null
  ./install.sh --run-cmake
  popd >/dev/null
  pushd tensorflow-cpp/tensorflow >/dev/null
  mkdir -p build && cd build
  cmake -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX="$TENSORFLOW_DIR" -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . -j"$JOBS" --target install
  popd >/dev/null
  rm -rf tensorflow-cpp
  popd >/dev/null
else
  echo "[AI] Tensorflow already present at $TENSORFLOW_DIR"
fi

# -------- CUDA Quantum (CPU path; we just need MLIR bits) --------
mkdir -p "$DEPS_DIR"
if [[ ! -d "$CUDAQ_DIR" ]]; then
  echo "[CUDAQ] Cloning CUDA Quantum to $CUDAQ_DIR"
  git clone https://github.com/NVIDIA/cuda-quantum.git "$CUDAQ_DIR"
else
  echo "[CUDAQ] CUDA Quantum already present at $CUDAQ_DIR"
fi

mkdir -p "$CUDAQ_DIR/build"
pushd "$CUDAQ_DIR/build" >/dev/null
echo "[CUDAQ] Configuring with $CMAKE_GENERATOR"
cmake -G "$CMAKE_GENERATOR" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DClang_DIR="$CLANG_DIR" \
  -DLLVM_DIR="$LLVM_DIR" \
  ..
echo "[CUDAQ] Building cudaq-mlir-runtime (-j$JOBS)"
cmake --build . -j"$JOBS" --target cudaq-mlir-runtime
popd >/dev/null

# -------- Your project --------
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" >/dev/null
echo "[MQSS] Configuring ($BUILD_TYPE)"
cmake -G "$CMAKE_GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH:-$HOME/.passes}" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_DIR="$LLVM_DIR" \
  -DClang_DIR="$CLANG_DIR" \
  -DBUILD_MLIR_PASSES_TOOLS=ON \
  -DBUILD_MLIR_PASSES_DOCS=OFF \
  -DBUILD_MLIR_PASSES_TESTS=ON \
  -DBUILD_MLIR_PASSES_AI=ON \
  -DCUDAQ_SOURCE_DIR="$CUDAQ_DIR" \
  ..

echo "[MQSS] Building (-j$JOBS)"
cmake --build . -j"$JOBS"

echo "[MQSS] (Optional) Install"
# cmake --install .

popd >/dev/null

echo "✅ Done."
