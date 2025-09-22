#!/bin/bash

# Clear the terminal screen
git config --global --add safe.directory '*'
clear

# Define directories
CURRENT_DIR=$(pwd)

INSTALL_PATH="${INSTALL_PATH:-$HOME/.passes}"
# Default values
NUM_JOBS=1  # Default number of jobs
BUILD_DOCS=OFF  # Default: Do not build documentation
BUILD_TESTS=ON  # Build tests by default
BUILD_TOOLS=ON  # Build tools by default
BUILD_AI=ON  # Build AI by default
BUILD_TYPE="Release"  # Default: Release mode

# Default directories (can be overridden by arguments)
MLIR_DIR="${MLIR_DIR:-/usr/local/llvm/lib/cmake/mlir}"
CLANG_DIR="${CLANG_DIR:-/usr/local/llvm/lib/cmake/clang}"
LLVM_DIR="${LLVM_DIR:-/usr/local/llvm/lib/cmake/llvm}"
INSTALL_DIR="${INSTALL_PATH:-$HOME/.passes}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--jobs)
      NUM_JOBS="$2"
      shift 2
      ;;
		--debug)
    	BUILD_TYPE="Debug"
    	shift
    	;;
    --mlir-dir)
      MLIR_DIR="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --clang-dir)
      CLANG_DIR="$2"
      shift 2
      ;;
    --llvm-dir)
      LLVM_DIR="$2"
      shift 2
      ;;
    --build-tools)
      BUILD_TOOLS=ON
      shift
      ;;
    --build-docs)
      BUILD_DOCS=ON
      shift
      ;;
    --build-tests)
      BUILD_TESTS=ON
      shift
      ;;
    --build-ai)
      BUILD_AI=ON
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

########################################################################################################################
# Build external tools necessary for the AI submodule
AI_DIR=${CURRENT_DIR}"/AI"
AI_EXTERNAL_DIR=${AI_DIR}"/external"
LIBTORCH_DIR=${AI_EXTERNAL_DIR}"/libtorch"
TENSORFLOW_DIR=${AI_EXTERNAL_DIR}"/tensorflow"
mkdir -p "${AI_EXTERNAL_DIR}"
if [ ! -d "${LIBTORCH_DIR}" ]; then
  cd "${AI_EXTERNAL_DIR}"
  wget https://download.pytorch.org/libtorch/nightly/cu129/libtorch-shared-with-deps-latest.zip
  unzip libtorch-shared-with-deps-latest.zip
  rm -rf libtorch-shared-with-deps-latest.zip
else
  echo "Libtorch already exists at ${LIBTORCH_DIR}."
fi
if [ ! -d "${TENSORFLOW_DIR}" ]; then
  cd "${AI_EXTERNAL_DIR}"
  git clone https://github.com/leggedrobotics/tensorflow-cpp.git
  cd tensorflow-cpp/eigen
  ./install.sh --run-cmake
  cd ../tensorflow
  mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX="${TENSORFLOW_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib ..
  make install -j
  cd "${AI_EXTERNAL_DIR}"
  rm -rf tensorflow-cpp
else
  echo "Tensorflow already exists at ${TENSORFLOW_DIR}."
fi
cd "${CURRENT_DIR}"
########################################################################################################################

BUILD_DIR=${CURRENT_DIR}"/build"
DEPS_DIR="${BUILD_DIR}/_deps"
CUDAQ_DIR="${DEPS_DIR}/cuda-quantum"
CUDAQ_REPO="https://github.com/NVIDIA/cuda-quantum.git"

# Create directories if they don't exist
mkdir -p "${BUILD_DIR}"
mkdir -p "${DEPS_DIR}"

# Clone the CUDA Quantum repository
echo "Cloning CUDA Quantum repository into ${CUDAQ_DIR}."
if [ ! -d "${CUDAQ_DIR}" ]; then
  git clone "${CUDAQ_REPO}" "${CUDAQ_DIR}"
  if [ $? -ne 0 ]; then
    echo "Failed to clone CUDA Quantum repository."
    exit 1
  fi
else
  echo "CUDA Quantum repository already exists at ${CUDAQ_DIR}. Skipping clone."
fi

# Navigate to the CUDA Quantum directory
cd "${CUDAQ_DIR}" || { echo "Failed to navigate to ${CUDAQ_DIR}."; exit 1; }

# Create a build directory
mkdir -p build && cd build || { echo "Failed to create or navigate to build directory."; exit 1; }

# Configure CUDA Quantum using CMake
echo "Configuring CUDA Quantum with CMake."
cmake -G Ninja \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DClang_DIR="${CLANG_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  ..

if [ $? -ne 0 ]; then
  echo "CMake configuration failed."
  exit 1
fi

# Build the cudaq-mlir-runtime target using Ninja
echo "Building cudaq-mlir-runtime target with ${NUM_JOBS} jobs."
ninja -j"${NUM_JOBS}" cudaq-mlir-runtime

if [ $? -ne 0 ]; then
  echo "Failed to build cudaq-mlir-runtime target."
  exit 1
fi

echo "Build completed successfully!"

echo ${BUILD_DIR}
cd  "${BUILD_DIR}" || { echo "Failed to navigate back to the original directory."; exit 1; }

echo "Configuring MQSS Passes Repository CMake."
cmake .. \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DBUILD_MLIR_PASSES_TOOLS="${BUILD_TOOLS}" \
  -DBUILD_MLIR_PASSES_DOCS="${BUILD_DOCS}" \
  -DBUILD_MLIR_PASSES_TESTS="${BUILD_TESTS}"\
  -DBUILD_MLIR_PASSES_AI="${BUILD_AI}" \
  -DCUDAQ_SOURCE_DIR="${CUDAQ_DIR}" \
	-DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
echo "Building MQSS Repository Passes with ${NUM_JOBS} jobs."
make -j"${NUM_JOBS}"
echo "Build of MQSS Repository Passes completed!"
