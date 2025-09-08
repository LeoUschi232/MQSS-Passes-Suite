# Project name and language specification
cmake_minimum_required(VERSION 3.19...3.30)
project(
        MLIRPasses
        VERSION 1.0
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_FLAGS "-w -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")

# Set default build type to Debug if none is specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
            Debug
            CACHE STRING "Build type (Debug, Release)" FORCE)
endif ()

# Define macros for Debug builds
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Configuring for Debug mode")
    add_compile_definitions(DEBUG)
endif ()

# doxygen dependencies
include(cmake/ExternalDependencies.cmake)
# ################ MLIR Settings ###########################################
# Find MLIR. MLIR is usually installed as part of an LLVM distribution.
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Append MLIR's CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Include the MLIR helper macros.
include(TableGen)
include(AddMLIR)
include(AddLLVM)
# Add LLVM, Clang and MLIR headers to the include path
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

# ############### ADDITIONAL PACKAGES #######################################
# Tell where to search for Find<package>.cmake files
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# set(CMAKE_CXX_STANDARD 11) Find QDMI package find_package(qdmi REQUIRED) Find
# QInfo package find_package(qinfo REQUIRED)
find_package(Z3 REQUIRED)

find_package(mqt-qcec REQUIRED)
# Find the QMAP package
set(CMAKE_CXX_STANDARD 17)
set(MQT_QMAP_TESTS OFF)
find_package(mqt-qmap REQUIRED)
# Restore the original CXX flags and compiler
set(CMAKE_CXX_STANDARD 20)
# ###################### CUDAQ settings ###################################
set(CUDAQ_MAIN_INCLUDE_DIR "${CUDAQ_SOURCE_DIR}/include")

set(CUDAQ_BINARY_DIR "${CUDAQ_SOURCE_DIR}/build")
set(CUDAQ_CMAKE_DIR "${CUDAQ_SOURCE_DIR}/cmake")
set(CUDAQ_INCLUDE_DIR "${CUDAQ_SOURCE_DIR}/build/include")
set(CUDAQ_LIBRARY_DIR "${CUDAQ_SOURCE_DIR}/build/lib")
set(CUDAQ_TOOLS_DIR "${CUDAQ_SOURCE_DIR}/build/bin")
set(CUDAQ_MLIR_RUNTIME_PATH "${CUDAQ_SOURCE_DIR}/build/lib")

include_directories(BEFORE ${CUDAQ_MAIN_INCLUDE_DIR}
        ${CUDAQ_BINARY_DIR}/include ${CUDAQ_SOURCE_DIR}/runtime)

# Validate CUDAQ dependencies
if (NOT EXISTS ${CUDAQ_BINARY_DIR}/lib/libcudaq-mlir-runtime.so)
    message(
            FATAL_ERROR
            "Invalid CUDAQ dir: ${CUDAQ_SOURCE_DIR}. Cannot find libcudaq.so in build directory."
    )
endif ()

link_directories(${CUDAQ_LIBRARY_DIR})
# ##############################################################################

set(MQT_QMAP_INCLUDE_DIR "${mqt-qmap_SOURCE_DIR}/include/")
set(MQT_QCEC_INCLUDE_DIR "${mqt-qcec_SOURCE_DIR}/include/")
set(MQT_CORE_INCLUDE_DIR "${mqt-core_SOURCE_DIR}/include/mqt-core")
set(qdmi_INCLUDE_DIRS "${QDMI_INCLUDE_DIRS}/qdmi")
set(qinfo_INCLUDE_DIRS "${qinfo_SOURCE_DIR}/include/qinfo")
message(STATUS "mqt include dir: ${MQT_QMAP_INCLUDE_DIR}")
message(STATUS "QINFO INCLUDE DIRS ${qinfo_INCLUDE_DIRS}")

include_directories(
        BEFORE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Passes/
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Support/
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Support/CodeGen/
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Support/DAG/
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Optimizer/
        ${MQT_QMAP_INCLUDE_DIR}
        ${MQT_CORE_INCLUDE_DIR}
        ${qdmi_INCLUDE_DIRS}
        ${qinfo_INCLUDE_DIRS})

# Set explicit paths
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/Passes)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/Optimizer)

# Generate .inc files from TableGen for Transforms.
set(LLVM_TARGET_DEFINITIONS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Passes/Transforms.td)
mlir_tablegen(include/Passes/Transforms.h.inc -gen-pass-decls -name
        MQSSOptTransforms)
add_public_tablegen_target(MQSSTransformsPassIncGen)

# Generate .inc files from  for Decompositions.
set(LLVM_TARGET_DEFINITIONS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Passes/Decompositions.td)
mlir_tablegen(include/Passes/Decompositions.h.inc -gen-pass-decls -name
        MQSSOptDecompositions)
add_public_tablegen_target(MQSSDecompositionsPassIncGen)

# Generate .inc files from TableGen for Cancellations.
set(LLVM_TARGET_DEFINITIONS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Passes/Cancellations.td)
mlir_tablegen(include/Passes/Cancellations.h.inc -gen-pass-decls -name
        MQSSOptCancellations)
add_public_tablegen_target(MQSSCancellationsPassIncGen)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/include)

# Collect all .cpp files in the src directory
# First transforms, decompositions and cancellations.
file(GLOB SRC_TRANSFORMS
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/Transforms/*.cpp")
file(GLOB SRC_DECOMPOSITIONS
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/Decompositions/*.cpp")
file(GLOB SRC_CANCELLATIONS
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/Cancellations/*.cpp")

# Then the other stuff.
file(GLOB SRC_CODEGEN "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/CodeGen/*.cpp")
file(GLOB SRC_EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/Examples/*.cpp")
file(GLOB SRC_OPTIMIZER "${CMAKE_CURRENT_SOURCE_DIR}/lib/Optimizer/*.cpp")
file(GLOB SRC_INTERFACES "${CMAKE_CURRENT_SOURCE_DIR}/lib/Interfaces/*.cpp")
file(GLOB_RECURSE SRC_SUPPORT_QUAKE "${CMAKE_CURRENT_SOURCE_DIR}/lib/Support/*.cpp")

add_mlir_library(
        SupportQuake
        SHARED
        ${SRC_SUPPORT_QUAKE}
        LINK_LIBS
        PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms)

add_mlir_library(
        Interfaces
        SHARED
        ${SRC_INTERFACES}
        LINK_LIBS
        PUBLIC
        SupportQuake
        MQT::CoreIR
        MQT::CoreNA
        MQT::QMapSC
        MLIRIR
        MLIRLinalgDialect
        MLIRLinalgUtils
        MLIRPass
        MLIRTransforms)

add_mlir_library(
        MLIRPasses
        SHARED
        ${SRC_TRANSFORMS}
        ${SRC_DECOMPOSITIONS}
        ${SRC_CANCELLATIONS}
        ${SRC_CODEGEN}
        ${SRC_EXAMPLES}
        DEPENDS
        MQSSTransformsPassIncGen
        MQSSDecompositionsPassIncGen
        MQSSCancellationsPassIncGen
        LINK_LIBS
        PUBLIC
        SupportQuake
        Interfaces
        MQT::CoreIR
        MQT::CoreNA
        MQT::QMapSC
        MQT::QMapSCHeuristic
        MLIRTensorDialect
        MLIRIR
        MLIRPass
        MLIRTransforms)

add_mlir_library(
        Optimizer
        SHARED
        ${SRC_OPTIMIZER}
        LINK_LIBS
        PUBLIC
        MLIRIR
        MLIRPasses)

target_compile_options(Interfaces PRIVATE -fexceptions -frtti)
target_compile_options(SupportQuake PRIVATE -fexceptions -frtti)
target_compile_options(MLIRPasses PRIVATE -fexceptions -frtti)

# Create an alias target for convenience
add_library(MQSS::MLIRPasses ALIAS MLIRPasses)
add_library(MQSS::SupportQuake ALIAS SupportQuake)
add_library(MQSS::Optimizer ALIAS Optimizer)
add_library(MQSS::Interfaces ALIAS Interfaces)

# ------------------------- AI targets (inlined) -------------------------
if (BUILD_MLIR_PASSES_AI)
    message(STATUS "Building AI utilities (Torch / TensorFlow selectors, converters)")

    set(AI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/AI")

    # Vendored SDK discovery roots
    list(APPEND CMAKE_PREFIX_PATH
            "${AI_DIR}/external/tensorflow"
            "${AI_DIR}/external/libtorch"
    )

    # Find each SDK (kept REQUIRED as in original AI/CMakeLists.txt)
    find_package(TensorFlow CONFIG REQUIRED)
    find_package(Torch REQUIRED)

    # ---------------- Shared (SDK-agnostic!) ----------------
    add_library(common_ai_props INTERFACE)
    target_include_directories(common_ai_props INTERFACE
            "${AI_DIR}/include"
    )
    target_compile_definitions(common_ai_props INTERFACE
            AI_DATASET_DIR="${AI_DIR}/Dataset"
            MQSS_BUILD_DIR="${CMAKE_BINARY_DIR}"
    )
    target_link_libraries(common_ai_props INTERFACE
            MLIRIR
            MLIRTransforms
            cudaq-mlir-runtime
            MLIRPasses
            SupportQuake
            Optimizer
            Interfaces
    )
    if (NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(common_ai_props INTERFACE -Ofast)
        if (NOT APPLE)
            target_compile_options(common_ai_props INTERFACE -march=native)
        endif ()
    endif ()

    # ---------------- Core utils (SDK-agnostic sources) ----------------
    file(GLOB UTILS_HEADERS CONFIGURE_DEPENDS "${AI_DIR}/include/Utils/*.hpp")
    file(GLOB UTILS_SOURCES CONFIGURE_DEPENDS "${AI_DIR}/src/Utils/*.cpp")

    add_library(ai_core STATIC ${UTILS_SOURCES} ${UTILS_HEADERS})
    target_link_libraries(ai_core PUBLIC common_ai_props)
    target_include_directories(ai_core PUBLIC "${AI_DIR}/include")
    target_include_directories(ai_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set_property(TARGET ai_core PROPERTY CXX_STANDARD 17)

    # ---------------- Torch-only targets ----------------
    file(GLOB_RECURSE TORCH_HEADERS CONFIGURE_DEPENDS "${AI_DIR}/include/Torch/*.hpp")
    file(GLOB_RECURSE TORCH_SOURCES CONFIGURE_DEPENDS "${AI_DIR}/src/Torch/*.cpp")

    if (TORCH_SOURCES OR TORCH_HEADERS)
        add_library(ai_torch STATIC ${TORCH_SOURCES} ${TORCH_HEADERS})
        target_link_libraries(ai_torch PUBLIC ai_core)
        if (TARGET Torch::Torch)
            target_link_libraries(ai_torch PUBLIC Torch::Torch)
        else ()
            target_link_libraries(ai_torch PUBLIC "${TORCH_LIBRARIES}")
            target_include_directories(ai_torch PUBLIC "${TORCH_INCLUDE_DIRS}")
        endif ()
        target_compile_options(ai_torch PRIVATE ${TORCH_CXX_FLAGS})
        target_include_directories(ai_torch PUBLIC "${AI_DIR}/include")
        target_include_directories(ai_torch PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        set_property(TARGET ai_torch PROPERTY CXX_STANDARD 17)
    endif ()

    add_executable(ai_pass_selector_torch "${AI_DIR}/ai_pass_selector_torch.cpp")
    set_property(TARGET ai_pass_selector_torch PROPERTY CXX_STANDARD 17)
    set_property(TARGET ai_pass_selector_torch PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(ai_pass_selector_torch PRIVATE ai_torch)

    # ---------------- TensorFlow-only targets ----------------
    file(GLOB_RECURSE TENSORFLOW_HEADERS CONFIGURE_DEPENDS "${AI_DIR}/include/Tensorflow/*.hpp")
    file(GLOB_RECURSE TENSORFLOW_SOURCES CONFIGURE_DEPENDS "${AI_DIR}/src/Tensorflow/*.cpp")

    if (TENSORFLOW_SOURCES OR TENSORFLOW_HEADERS)
        add_library(ai_tensorflow STATIC ${TENSORFLOW_SOURCES} ${TENSORFLOW_HEADERS})
        target_link_libraries(ai_tensorflow PUBLIC ai_core)
        target_link_libraries(ai_tensorflow PUBLIC TensorFlow::TensorFlow)
        target_include_directories(ai_tensorflow PUBLIC "${AI_DIR}/include")
        target_include_directories(ai_tensorflow PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        set_property(TARGET ai_tensorflow PROPERTY CXX_STANDARD 17)
    endif ()

    add_executable(ai_pass_selector_tensorflow "${AI_DIR}/ai_pass_selector_tensorflow.cpp")
    set_property(TARGET ai_pass_selector_tensorflow PROPERTY CXX_STANDARD 17)
    set_property(TARGET ai_pass_selector_tensorflow PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(ai_pass_selector_tensorflow PRIVATE ai_tensorflow)

    # ---------------- Neutral executables (no TF/Torch) ----------------
    add_executable(convert_qasm_dataset_to_quake "${AI_DIR}/convert_qasm_dataset_to_quake.cpp")
    set_property(TARGET convert_qasm_dataset_to_quake PROPERTY CXX_STANDARD 17)
    set_property(TARGET convert_qasm_dataset_to_quake PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(convert_qasm_dataset_to_quake PRIVATE ai_core)

    add_executable(convert_quake_passtest_to_tikz "${AI_DIR}/convert_quake_passtest_to_tikz.cpp")
    set_property(TARGET convert_quake_passtest_to_tikz PROPERTY CXX_STANDARD 17)
    set_property(TARGET convert_quake_passtest_to_tikz PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(convert_quake_passtest_to_tikz PRIVATE ai_core)
endif ()
# -----------------------------------------------------------------------

# add tests
if (BUILD_MLIR_PASSES_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif ()

# add documentation
if (BUILD_MLIR_PASSES_DOCS)
    add_subdirectory(docs)
endif ()

if (BUILD_MLIR_PASSES_TOOLS)
    add_subdirectory(tools)
endif ()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)

# Enable FetchContent for integration
include(FetchContent)