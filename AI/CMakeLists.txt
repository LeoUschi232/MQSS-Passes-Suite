cmake_minimum_required(VERSION 3.19)
project(AIPassSelector LANGUAGES CXX)

# Vendored SDK discovery roots
list(APPEND CMAKE_PREFIX_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tensorflow"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch"
)

# Find each SDK, but we will not mix them in a single target
find_package(TensorFlow CONFIG REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- Shared (SDK-agnostic!) ----------------
add_library(common_ai_props INTERFACE)
target_include_directories(common_ai_props INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_definitions(common_ai_props INTERFACE
        AI_DATASET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Dataset"
        AI_AGENTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Agents"
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

# Gather sources (no SDK headers/sources here)
file(GLOB_RECURSE ENV_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/Environment/*.hpp")
file(GLOB_RECURSE ENV_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Environment/*.cpp")
file(GLOB UTILS_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/Utils/*.hpp")
file(GLOB UTILS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/*.cpp")

add_library(ai_core STATIC ${UTILS_HEADERS} ${UTILS_SOURCES} ${ENV_HEADERS} ${ENV_SOURCES})
target_link_libraries(ai_core PUBLIC common_ai_props)
target_include_directories(ai_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(ai_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set_property(TARGET ai_core PROPERTY CXX_STANDARD 17)

# ---------------- Torch-only targets ----------------
file(GLOB_RECURSE TORCH_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/Torch/*.hpp")
file(GLOB_RECURSE TORCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Torch/*.cpp")

if (TORCH_SOURCES OR TORCH_HEADERS)
    add_library(ai_torch STATIC ${TORCH_SOURCES} ${TORCH_HEADERS})
    target_link_libraries(ai_torch PUBLIC ai_core)
    # Link ONLY Torch here
    if (TARGET Torch::Torch)
        target_link_libraries(ai_torch PUBLIC Torch::Torch)
    else ()
        target_link_libraries(ai_torch PUBLIC "${TORCH_LIBRARIES}")
        target_include_directories(ai_torch PUBLIC "${TORCH_INCLUDE_DIRS}")
    endif ()
    target_compile_options(ai_torch PRIVATE ${TORCH_CXX_FLAGS})
    target_include_directories(ai_torch PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_include_directories(ai_torch PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    set_property(TARGET ai_torch PROPERTY CXX_STANDARD 17)
endif ()

add_executable(ai_pass_selector_torch ai_pass_selector_torch.cpp)
target_link_libraries(ai_pass_selector_torch PRIVATE ai_torch)

# ---------------- TensorFlow-only targets ----------------
file(GLOB_RECURSE TENSORFLOW_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/Tensorflow/*.hpp")
file(GLOB_RECURSE TENSORFLOW_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Tensorflow/*.cpp")

if (TENSORFLOW_SOURCES OR TENSORFLOW_HEADERS)
    add_library(ai_tensorflow STATIC ${TENSORFLOW_SOURCES} ${TENSORFLOW_HEADERS})
    target_link_libraries(ai_tensorflow PUBLIC ai_core)
    # Link ONLY TensorFlow here
    target_link_libraries(ai_tensorflow PUBLIC TensorFlow::TensorFlow)
    target_include_directories(ai_tensorflow PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_include_directories(ai_tensorflow PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    set_property(TARGET ai_tensorflow PROPERTY CXX_STANDARD 17)
endif ()

add_executable(ai_pass_selector_tensorflow ai_pass_selector_tensorflow.cpp)
target_link_libraries(ai_pass_selector_tensorflow PRIVATE ai_tensorflow)

# ---------------- Neutral executables (no TF/Torch) ----------------
add_executable(convert_qasm_dataset_to_quake convert_qasm_dataset_to_quake.cpp)
target_link_libraries(convert_qasm_dataset_to_quake PRIVATE ai_core)

add_executable(convert_quake_testsuite_to_tikz convert_quake_testsuite_to_tikz.cpp)
target_link_libraries(convert_quake_testsuite_to_tikz PRIVATE ai_core)
