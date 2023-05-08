#
# File: test/CMakeLists.txt

# These tests are all Makefile based tests.
# Each test is contained in its own subdirectory of this test directory
# The Makefile must create a binary with the same name as the directory name.

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(test-hipfort)
endif()

set(CMAKE_BUILD_PARALLEL_LEVEL "4" CACHE STRING "Number of build jobs")

add_custom_target(all-f2003-tests
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2003 build)
add_custom_target(all-f2003-tests-run
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j1 PREFIX=f2003 run
	DEPENDS all-f2003-tests)
add_custom_target(all-f2003-tests-clean
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2003 clean)

add_custom_target(all-f2008-tests
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2008 build)
add_custom_target(all-f2008-tests-run
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j1 PREFIX=f2008 run
	DEPENDS all-f2008-tests)
add_custom_target(all-f2008-tests-clean
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2008 clean)

add_custom_target(all-tests
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} build_all)
add_custom_target(all-tests-run
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j1 run_all
	DEPENDS all-tests)
add_custom_target(all-tests-clean
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} clean_all)
