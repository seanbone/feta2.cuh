cmake_minimum_required(VERSION 3.20)
project(feta2_tests LANGUAGES CUDA CXX)

# Supress a warning in newer CMake versions when using FetchContent
if (POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()


#### Dependency: GoogleTest

include(FetchContent)
message(STATUS "* Downloading dependency: GoogleTest")
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
message(STATUS "* Downloading dependency: GoogleTest - done")

include(GoogleTest)


#### Add unit tests

# Treat files named test_*.cu in current directory as tests
file(GLOB TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu
)
list(SORT TEST_FILES COMPARE NATURAL)

foreach(file ${TEST_FILES})
    get_filename_component(filename ${file} NAME_WE)
    message(STATUS "* Found unit test file " ${file})
endforeach()

list(LENGTH TEST_FILES TEST_FILES_LEN)
message(STATUS "* Found " ${TEST_FILES_LEN} " unit test files")

add_executable(feta2_tests ${TEST_FILES})
target_link_libraries(feta2_tests feta2 GTest::gtest_main)
# TODO figure out better way to do this
target_compile_options(feta2_tests PRIVATE "--diag-suppress=20012")
target_include_directories(feta2_tests PRIVATE include)
gtest_discover_tests(feta2_tests)
