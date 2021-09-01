cmake_minimum_required(VERSION 3.10)

option(DEBUG "If ON, executable will try to diagnose where failures occur." OFF)

# set compilers
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

# set the project name
project(DistributedSBP VERSION 2021.08.29)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(extern/hungarian)

set(INCLUDE_DIRS
    include
    include/blockmodel
    include/blockmodel/sparse
    extern/tclap-1.2.2/include
    patches
    extern/hungarian)

# add the executable
add_library(SBP SHARED
    graph.cpp
    utils.cpp
    sbp.cpp
    evaluate.cpp
    common.cpp
    finetune.cpp
    block_merge.cpp
    partition.cpp
    blockmodel/blockmodel.cpp
    blockmodel/blockmodel_triplet.cpp
    blockmodel/dist_blockmodel.cpp
    blockmodel/sparse/dict_matrix.cpp
    blockmodel/sparse/dict_transpose_matrix.cpp
    blockmodel/sparse/dist_dict_matrix.cpp)
target_include_directories(SBP PUBLIC ${INCLUDE_DIRS})

add_executable(DistributedSBP main.cpp)
target_include_directories(DistributedSBP PUBLIC ${INCLUDE_DIRS})
#
#target_sources(DistributedSBP PUBLIC
#    # main.cpp
#    graph.cpp
#    utils.cpp
#    sbp.cpp
#    evaluate.cpp
#    common.cpp
#    finetune.cpp
#    block_merge.cpp
#    partition.cpp
#    blockmodel/blockmodel.cpp
#    blockmodel/blockmodel_triplet.cpp
#    blockmodel/dist_blockmodel.cpp
#    blockmodel/sparse/dict_matrix.cpp
#    blockmodel/sparse/dict_transpose_matrix.cpp
#    blockmodel/sparse/dist_dict_matrix.cpp)

if(DEBUG)
    target_compile_options(DistributedSBP PUBLIC -Wall -O3 -no-pie -fopenmp -g3 -fsanitize=address)
    target_link_libraries(DistributedSBP PUBLIC SBP hungarian -fopenmp stdc++fs -fsanitize=address)
else(DEBUG)
#    target_compile_options(DistributedSBP PUBLIC -Wall -O3 -no-pie -fopenmp)
#    target_link_libraries(DistributedSBP PUBLIC SBP hungarian -fopenmp stdc++fs)
    target_compile_options(SBP PUBLIC -Wall -O3 -no-pie -fopenmp)
    target_link_libraries(SBP PUBLIC hungarian -fopenmp stdc++fs)
    target_link_libraries(DistributedSBP PUBLIC SBP)
endif(DEBUG)

# Google Test Suite
enable_testing()
add_subdirectory(extern/googletest)
add_executable(Test test/finetune_test.cpp)
target_include_directories(Test PUBLIC ${INCLUDE_DIRS})
#target_sources(Test PUBLIC test/finetune_test.cpp)
include(GoogleTest)
target_link_libraries(Test SBP gtest_main hungarian -fopenmp stdc++fs)
gtest_discover_tests(Test)
#find_package(GTest REQUIRED)
#include(FetchContent)
#FetchContent_Declare(
#    googletest
#    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
#)
