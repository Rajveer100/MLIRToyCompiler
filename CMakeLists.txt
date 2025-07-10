cmake_minimum_required(VERSION 3.28)
project(MiniToyCompiler)

# Set C/C++ compiler
set(CMAKE_C_COMPILER "/Users/rajveersingh/GitHub-OpenSource/llvm-standalone-dev/llvm-project/build/bin/clang")
set(CMAKE_CXX_COMPILER "/Users/rajveersingh/GitHub-OpenSource/llvm-standalone-dev/llvm-project/build/bin/clang++")

set(CC "/Users/rajveersingh/GitHub-OpenSource/llvm-standalone-dev/llvm-project/build/bin/clang")
set(CXX "/Users/rajveersingh/GitHub-OpenSource/llvm-standalone-dev/llvm-project/build/bin/clang++")

# Set C++ standard
set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE DEBUG)

# Set cmake export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set LLVM and MLIR paths
set(LLVM_DIR "~/GitHub-OpenSource/llvm-standalone-dev/llvm-project/build/lib/cmake/llvm/")
set(MLIR_DIR "~/GitHub-OpenSource/llvm-standalone-dev/llvm-project/build/lib/cmake/mlir/")

# Find package LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Set LLVM/MLIR runtime output directories and binary directories.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Append LLVM/MLIR paths to CMake modules path.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include TableGen/AddLLVM/AddMLIR/HandleLLVMOptions
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include LLVM/MLIR directories.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Include Project source and binary directories.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Link LLVM build libraries and add definitions.
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
