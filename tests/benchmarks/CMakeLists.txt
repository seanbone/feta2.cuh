cmake_minimum_required(VERSION 3.20)
project(feta2_benchmarks LANGUAGES CUDA CXX)

# Treat files named bench_*.cu in current directory as tests
file(GLOB BENCH_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_*.cu
)
list(SORT BENCH_FILES COMPARE NATURAL)

foreach(file ${BENCH_FILES})
  get_filename_component(filename ${file} NAME_WE)
  message(STATUS "* Found benchmark file " ${file})
endforeach()

list(LENGTH BENCH_FILES BENCH_FILES_LEN)
message(STATUS "* Found " ${BENCH_FILES_LEN} " benchmark files")

add_executable(feta2_benchmarks ${BENCH_FILES})
target_link_libraries(feta2_benchmarks feta2 GTest::gtest_main)
target_include_directories(feta2_benchmarks PRIVATE include)
target_compile_options(feta2_benchmarks PRIVATE ${FETA2_COMMON_OPTIONS})
gtest_discover_tests(feta2_benchmarks)

