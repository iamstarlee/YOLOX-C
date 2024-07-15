cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(ort_utility VERSION 0.0.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(onnxruntime_INSTALL_PREFIX /usr/local)
set(onnxruntime_INCLUDE_DIRS
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
)

find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)

find_package(CUDA QUIET)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Find CUDA !")
else()
    message(FATAL_ERROR "Not found CUDA!")
endif()


if(NOT DEFINED USE_GPU)
  set(USE_GPU ON)
endif()

message(STATUS "USE_GPU: ${USE_GPU}")

if(CUDA_FOUND AND USE_GPU)
  add_definitions(-DENABLE_GPU=1)
else()
  add_definitions(-DENABLE_GPU=0)
endif()


add_compile_options(
  "$<$<CONFIG:Debug>:-DENABLE_DEBUG=1>"
  "$<$<CONFIG:Release>:-DENABLE_DEBUG=0>"
)

add_subdirectory(src)
add_subdirectory(examples)

# cmake通过指定环境变量能打印出编译和链接阶段的耗时情况
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -ggdb -Og -fPIC -w -Wl,--export-dynamic -Wno-deprecated -fpermissive -save-temps")
