cmake_minimum_required(VERSION 3.16)

project(QuarkLA VERSION 1.0.0)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

# Compiler warnings (after external libraries)
set(WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
)

# Header-only library
add_library(QuarkLALib INTERFACE)
target_include_directories(QuarkLALib INTERFACE include)

# Main executable
add_executable(QuarkLA src/main.cpp)
target_link_libraries(QuarkLA QuarkLALib)
target_compile_options(QuarkLA PRIVATE ${WARNING_FLAGS})

# Enable testing
enable_testing()

# Vec tests
add_executable(test_vec tests/test_vec.cpp)
target_link_libraries(test_vec QuarkLALib gtest gtest_main)
target_compile_options(test_vec PRIVATE ${WARNING_FLAGS})

# All tests
add_executable(test_all 
    tests/test_vec.cpp
)
target_link_libraries(test_all QuarkLALib gtest gtest_main)
target_compile_options(test_all PRIVATE ${WARNING_FLAGS})

# Add tests to CTest
add_test(NAME VecTests COMMAND test_vec)
add_test(NAME AllTests COMMAND test_all)

# Print build information
message(STATUS "")
message(STATUS "QuarkLA Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")