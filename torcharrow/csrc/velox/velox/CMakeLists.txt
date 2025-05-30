# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# set the project name
project(velox)

# Include our ThirdPartyToolchain dependencies macros
include(ThirdpartyToolchain.cmake)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add all options below
option(
  VELOX_BUILD_TESTING
  "Enable Velox tests. This will enable all other build options automatically."
  ON)
option(
  VELOX_BUILD_MINIMAL
  "Build a minimal set of components only. This will override other build options."
  OFF)

option(VELOX_ENABLE_DUCKDB "Build duckDB to enable differential testing." ON)
option(VELOX_ENABLE_EXEC "Build exec." ON)
option(VELOX_ENABLE_AGGREGATES "Build aggregates." ON)
option(VELOX_ENABLE_HIVE_CONNECTOR "Build Hive connector." ON)
option(VELOX_ENABLE_TPCH_CONNECTOR "Build TPC-H connector." ON)
option(VELOX_ENABLE_PRESTO_FUNCTIONS "Build Presto SQL functions." ON)
option(VELOX_ENABLE_SPARK_FUNCTIONS "Build Spark SQL functions." ON)
option(VELOX_ENABLE_EXPRESSION "Build expression." ON)
option(VELOX_ENABLE_PARSE "Build parser used for unit tests." ON)
option(VELOX_ENABLE_EXAMPLES
       "Build examples. This will enable VELOX_ENABLE_EXPRESSION automatically."
       OFF)
option(VELOX_ENABLE_SUBSTRAIT "Buid Substrait-to-Velox converter." OFF)
option(VELOX_ENABLE_BENCHMARKS "Build velox top level benchmarks." OFF)
option(VELOX_ENABLE_BENCHMARKS_BASIC "Build velox basic benchmarks." OFF)
option(VELOX_ENABLE_S3 "Build S3 Connector" OFF)
option(VELOX_ENABLE_HDFS "Build Hdfs Connector" OFF)
option(VELOX_ENABLE_PARQUET "Enable Parquet support" OFF)
option(VELOX_ENABLE_ARROW "Enable Arrow support" OFF)
option(VELOX_BUILD_TEST_UTILS "Enable Velox test utilities" OFF)

if(${VELOX_BUILD_MINIMAL})
  # Enable and disable components for velox base build
  set(VELOX_BUILD_TESTING OFF)
  set(VELOX_ENABLE_PRESTO_FUNCTIONS ON)
  set(VELOX_ENABLE_DUCKDB OFF)
  set(VELOX_ENABLE_EXPRESSION ON)
  set(VELOX_ENABLE_PARSE OFF)
  set(VELOX_ENABLE_EXEC OFF)
  set(VELOX_ENABLE_AGGREGATES OFF)
  set(VELOX_ENABLE_HIVE_CONNECTOR OFF)
  set(VELOX_ENABLE_TPCH_CONNECTOR OFF)
  set(VELOX_ENABLE_SPARK_FUNCTIONS OFF)
  set(VELOX_ENABLE_EXAMPLES OFF)
  set(VELOX_ENABLE_S3 OFF)
  set(VELOX_ENABLE_SUBSTRAIT OFF)
  set(VELOX_CODEGEN_SUPPORT OFF)
endif()

if(${VELOX_BUILD_TESTING})
  # Enable all components to build testing binaries
  set(VELOX_ENABLE_PRESTO_FUNCTIONS ON)
  set(VELOX_ENABLE_DUCKDB ON)
  set(VELOX_ENABLE_EXPRESSION ON)
  set(VELOX_ENABLE_PARSE ON)
  set(VELOX_ENABLE_EXEC ON)
  set(VELOX_ENABLE_AGGREGATES ON)
  set(VELOX_ENABLE_HIVE_CONNECTOR ON)
  set(VELOX_ENABLE_TPCH_CONNECTOR ON)
  set(VELOX_ENABLE_SPARK_FUNCTIONS ON)
  set(VELOX_ENABLE_TEST_UTILS OFF)
  set(VELOX_ENABLE_EXAMPLES ON)
endif()

if(${VELOX_ENABLE_EXAMPLES})
  set(VELOX_ENABLE_EXPRESSION ON)
  set(VELOX_ENABLE_TEST_UTILS ON)
endif()

if(${VELOX_ENABLE_BENCHMARKS} OR ${VELOX_ENABLE_BENCHMARKS_BASIC})
  set(VELOX_ENABLE_DUCKDB ON)
  set(VELOX_ENABLE_PARSE ON)
  set(VELOX_BUILD_TEST_UTILS ON)
endif()

if(VELOX_ENABLE_S3)
  # Set AWS_ROOT_DIR if you have a custom install location of AWS SDK CPP.
  if(AWSSDK_ROOT_DIR)
    set(CMAKE_PREFIX_PATH ${AWSSDK_ROOT_DIR})
  endif()
  find_package(AWSSDK REQUIRED COMPONENTS s3;identity-management)
  add_definitions(-DVELOX_ENABLE_S3)
endif()

if(VELOX_ENABLE_HDFS)
  find_library(
    LIBHDFS3
    NAMES libhdfs3.so libhdfs3.dylib
    HINTS "${CMAKE_SOURCE_DIR}/hawq/depends/libhdfs3/_build/src/" REQUIRED)
  add_definitions(-DVELOX_ENABLE_HDFS3)
endif()

if(VELOX_ENABLE_PARQUET)
  add_definitions(-DVELOX_ENABLE_PARQUET)
  # Native Parquet reader requires Apache Thrift and Arrow Parquet writer, which
  # are included in Arrow.
  set(VELOX_ENABLE_ARROW ON)
endif()

# Turn on Codegen only for Clang and non Mac systems.
if((NOT DEFINED VELOX_CODEGEN_SUPPORT)
   AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   AND NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  message(STATUS "Enabling Codegen")
  set(VELOX_CODEGEN_SUPPORT True)
else()
  message(STATUS "Disabling Codegen")
  set(VELOX_CODEGEN_SUPPORT False)
endif()

# define processor variable for conditional compilation
if(${VELOX_CODEGEN_SUPPORT})
  add_compile_definitions(CODEGEN_ENABLED=1)
endif()

# MacOSX enables two-level namespace by default:
# http://mirror.informatimago.com/next/developer.apple.com/releasenotes/DeveloperTools/TwoLevelNamespaces.html
# Enables -flat_namespace so type_info can be deudplicated across .so boundaries
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_link_options("-Wl,-flat_namespace")
endif()

if(UNIX AND NOT APPLE)
  # codegen linker flags, -export-dynamic for rtti
  add_link_options("-Wl,-export-dynamic")
endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64" AND ${CMAKE_SYSTEM_NAME} MATCHES
                                                 "Darwin")
  set(ON_APPLE_M1 True)
endif()

# Required so velox code can be used in a dynamic library
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

execute_process(
  COMMAND
    bash -c
    "( source ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup-helper-functions.sh && echo -n $(get_cxx_flags $ENV{CPU_TARGET}))"
  OUTPUT_VARIABLE SCRIPT_CXX_FLAGS
  RESULT_VARIABLE COMMAND_STATUS)

if(COMMAND_STATUS EQUAL "1")
  message(FATAL_ERROR "Unable to determine compiler flags!")
endif()
message("Setting CMAKE_CXX_FLAGS=${SCRIPT_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCRIPT_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D USE_VELOX_COMMON_BASE")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAS_UNCAUGHT_EXCEPTIONS")

# Under Ninja, we are able to designate certain targets large enough to require
# restricted parallelism.
if("${MAX_HIGH_MEM_JOBS}")
  set_property(GLOBAL PROPERTY JOB_POOLS
                               "high_memory_pool=${MAX_HIGH_MEM_JOBS}")
else()
  set_property(GLOBAL PROPERTY JOB_POOLS high_memory_pool=1000)
endif()

if("${MAX_LINK_JOBS}")
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS
                                      "link_job_pool=${MAX_LINK_JOBS}")
  set(CMAKE_JOB_POOL_LINK link_job_pool)
endif()

if("${TREAT_WARNINGS_AS_ERRORS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if("${ENABLE_ALL_WARNINGS}")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(KNOWN_COMPILER_SPECIFIC_WARNINGS
        "-Wno-range-loop-analysis -Wno-mismatched-tags")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(KNOWN_COMPILER_SPECIFIC_WARNINGS
        "-Wno-implicit-fallthrough \
         -Wno-empty-body \
         -Wno-class-memaccess \
         -Wno-comment \
         -Wno-int-in-bool-context \
         -Wno-redundant-move \
         -Wno-array-bounds \
         -Wno-maybe-uninitialized \
         -Wno-unused-result \
         -Wno-format-overflow \
         -Wno-strict-aliasing \
         -Wno-type-limits")
  endif()

  set(KNOWN_WARNINGS
      "-Wno-unused \
       -Wno-unused-parameter \
       -Wno-sign-compare \
       -Wno-ignored-qualifiers \
       ${KNOWN_COMPILER_SPECIFIC_WARNINGS}")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ${KNOWN_WARNINGS}")
endif()

message("FINAL CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Boost_USE_MULTITHREADED TRUE)
find_package(
  Boost
  1.66.0
  REQUIRED
  program_options
  context
  filesystem
  regex
  thread
  system
  date_time
  atomic)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# Range-v3 will be enable when the codegen code actually lands keeping it here
# for reference. find_package(range-v3)

find_package(gflags COMPONENTS shared)
find_package(glog REQUIRED)

find_library(FMT fmt)

find_library(EVENT event)

find_library(DOUBLE_CONVERSION double-conversion)

if(NOT ${VELOX_BUILD_MINIMAL})
  find_library(LZ4 lz4)
  find_library(LZO lzo2)
  find_library(ZSTD zstd)
  find_package(ZLIB)
  find_library(SNAPPY snappy)
endif()

find_library(RE2 re2 REQUIRED)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  if(ON_APPLE_M1)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/icu4c" ${CMAKE_PREFIX_PATH})
  else()
    set(CMAKE_PREFIX_PATH "/usr/local/opt/icu4c" ${CMAKE_PREFIX_PATH})
  endif()
  find_package(ICU REQUIRED)
  include_directories(${ICU_INCLUDE_DIRS})
  link_directories("${ICU_INCLUDE_DIRS}/../lib")
endif()

set(folly_SOURCE AUTO)
resolve_dependency(folly)

set(FOLLY_WITH_DEPENDENCIES
    ${FOLLY_LIBRARIES}
    ${Boost_LIBRARIES}
    ${DOUBLE_CONVERSION}
    ${EVENT}
    ${SNAPPY}
    ${CMAKE_DL_LIBS}
    ${FMT})

if(DEFINED FOLLY_BENCHMARK_STATIC_LIB)
  set(FOLLY_BENCHMARK ${FOLLY_BENCHMARK_STATIC_LIB})
else()
  set(FOLLY_BENCHMARK Folly::follybenchmark)
endif()

if(NOT ${VELOX_BUILD_MINIMAL})
  find_package(BZip2 MODULE)
  if(BZIP2_FOUND)
    list(APPEND FOLLY_WITH_DEPENDENCIES ${BZIP2_LIBRARIES})
  endif()
endif()

include_directories(SYSTEM ${FOLLY_INCLUDE_DIRS})

if(NOT ${VELOX_BUILD_MINIMAL})
  find_package(Protobuf 3.0.0 REQUIRED)
endif()

# GCC needs to link a library to enable std::filesystem.
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(FILESYSTEM "stdc++fs")

  # Ensure we have gcc at least 8+.
  if(CMAKE_CXX_COMPILER_VERSION LESS 8.0)
    message(
      FATAL_ERROR "VELOX requires gcc > 8. Found ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
else()
  set(FILESYSTEM "")
endif()

if(VELOX_BUILD_TESTING AND NOT VELOX_ENABLE_DUCKDB)
  message(
    FATAL_ERROR
      "Unit tests require duckDB to be enabled (VELOX_ENABLE_DUCKDB=ON or VELOX_BUILD_TESTING=OFF)"
  )
endif()

set(VELOX_DISABLE_GOOGLETEST OFF)
if(NOT VELOX_BUILD_TEST_UTILS AND NOT VELOX_BUILD_TESTING)
  set(VELOX_DISABLE_GOOGLETEST ON)
  add_definitions(-DVELOX_DISABLE_GOOGLETEST)
endif()

# On macOS, search Homebrew for keg-only versions of Bison and Flex. Xcode does
# not provide new enough versions for us to use.
if(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
  execute_process(
    COMMAND brew --prefix bison
    RESULT_VARIABLE BREW_BISON
    OUTPUT_VARIABLE BREW_BISON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(BREW_BISON EQUAL 0 AND EXISTS "${BREW_BISON_PREFIX}")
    message(
      STATUS "Found Bison keg installed by Homebrew at ${BREW_BISON_PREFIX}")
    set(BISON_EXECUTABLE "${BREW_BISON_PREFIX}/bin/bison")
  endif()

  execute_process(
    COMMAND brew --prefix flex
    RESULT_VARIABLE BREW_FLEX
    OUTPUT_VARIABLE BREW_FLEX_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(BREW_FLEX EQUAL 0 AND EXISTS "${BREW_FLEX_PREFIX}")
    message(
      STATUS "Found Flex keg installed by Homebrew at ${BREW_FLEX_PREFIX}")
    set(FLEX_EXECUTABLE "${BREW_FLEX_PREFIX}/bin/flex")
    set(FLEX_INCLUDE_DIR "${BREW_FLEX_PREFIX}/include")
  endif()
endif()
find_package(BISON 3.0.4 REQUIRED)
find_package(FLEX 2.5.13 REQUIRED)

include_directories(SYSTEM velox)
include_directories(SYSTEM velox/external)
include_directories(SYSTEM velox/external/duckdb)
include_directories(SYSTEM velox/external/duckdb/tpch/dbgen/include)

include(CTest) # include after project() but before add_subdirectory()

include_directories(.)

# TODO: Include all other installation files. For now just making sure this
# generates an installable makefile.
install(FILES velox/type/Type.h DESTINATION "include/velox")

add_subdirectory(third_party)
add_subdirectory(velox)
