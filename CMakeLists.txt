cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

#set(CMAKE_GENERATOR_TOOLSET "ClangCL")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FGAC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CUDA_ARCHITECTURES 75)

project(fgac VERSION 0.1.0)
enable_language(CUDA)

# source file
file(GLOB_RECURSE HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/source/*.h")
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
file(GLOB_RECURSE CUH_FILES "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cuh")
file(GLOB_RECURSE CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Source" FILES ${HEADER_FILES} ${SOURCE_FILES} ${CUH_FILES} ${CU_FILES})

include_directories(${FGAC_ROOT_DIR}/source)
include_directories(${FGAC_ROOT_DIR}/include)

# thirdparty
include_directories(${FGAC_ROOT_DIR}/thirdparty/spdlog)
include_directories(${FGAC_ROOT_DIR}/thirdparty/stb)

add_definitions(-DFGAC_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# executable
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${CUH_FILES} ${CU_FILES})
