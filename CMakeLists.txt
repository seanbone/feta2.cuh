cmake_minimum_required(VERSION 3.20)

project(feta2 VERSION 2.0.0 LANGUAGES CUDA CXX)


#### OPTIONS

option(FETA2_BUILD_TESTS "Build unit tests" OFF)
option(FETA2_BUILD_BENCHMARKS "Build benchmarks. Only valid with FETA2_DEBUG_MODE=OFF" OFF)
option(FETA2_BUILD_DEMO "Build the demo" OFF)
option(FETA2_DEBUG_MODE "Compile with debugging features" OFF)

option(FETA2_WARN_AS_ERR "Treat compiler warnings as errors" OFF)
mark_as_advanced(FETA2_WARN_AS_ERR)

option(FETA2_NOT_FETA "Prevent aliasing feta2 as feta" OFF)
mark_as_advanced(FETA2_NOT_FETA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


### DEPENDENCIES

# Require Eigen >= 3.3
find_package(Eigen3 3.3 REQUIRED CONFIG)
message(STATUS "* Found Eigen version ${Eigen3_VERSION}")


#### LIBRARY

add_library(feta2 INTERFACE)
target_include_directories(feta2 INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(feta2 INTERFACE Eigen3::Eigen)

# Require C++17 features
target_compile_features(feta2 INTERFACE cxx_std_17)

if (FETA2_DEBUG_MODE)
  target_compile_definitions(feta2 INTERFACE FETA2_DEBUG_MODE)
endif()

if (FETA2_NOT_FETA)
  target_compile_definitions(feta2 INTERFACE FETA2_NOT_FETA)
endif()

if (FETA2_WARN_AS_ERR)
  target_compile_options(
    feta2 INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>
  )
  target_compile_options(
    feta2 INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
  )
endif()

# Allows calling constexpr __host__ functions from __device__ functions
# Otherwise this causes a slew of warnings from Eigen's code
target_compile_options(feta2 INTERFACE "--expt-relaxed-constexpr")
target_compile_options(feta2 INTERFACE "-Wno-unknown-pragmas")


#### DEMO EXECUTABLE

if (FETA2_BUILD_DEMO)
  add_executable(feta2_demo feta2_demo.cu)
  target_link_libraries(feta2_demo feta2)
  target_compile_options(feta2_demo PRIVATE
    "-Wall -Wextra -O3 -ftemplate-backtrace-limit=100"
  )
  # TODO figure out better way to do this
  target_compile_options(feta2_demo PRIVATE "--diag-suppress=20012")
endif()


#### UNIT TESTS

if (FETA2_BUILD_TESTS)
  add_subdirectory(tests)
endif()


#### INSTALLATION

include(GNUInstallDirs)

# TODO: only export install target if this is the top-level project

