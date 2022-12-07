cmake_minimum_required(VERSION 3.10)

option(DEBUG "If ON, executable will try to diagnose where failures occur." OFF)
option(SCOREP "If ON, will add score-p instrumentation to the executable." OFF)

# set compilers
if (SCOREP)
    set(CMAKE_C_COMPILER scorep-mpicc)
    set(CMAKE_CXX_COMPILER scorep-mpicxx)
else (SCOREP)
    set(CMAKE_C_COMPILER mpicc)
    set(CMAKE_CXX_COMPILER mpic++)
endif (SCOREP)

# set the project name
project(DistributedSBP VERSION 2022.02.04)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(extern/hungarian)
add_subdirectory(extern/abseil-cpp)
add_subdirectory(extern/robin-map)

set(INCLUDE_DIRS
        include
        include/blockmodel
        include/blockmodel/sparse
        include/distributed
        extern/tclap-1.2.2/include
        patches
        extern/hungarian
        extern/abseil-cpp
        extern/robin-map/include
        )

# add the executable
add_library(SBP STATIC
        src/graph.cpp
        src/utils.cpp
        src/sbp.cpp
        src/entropy.cpp
        src/evaluate.cpp
        src/common.cpp
        src/finetune.cpp
        src/block_merge.cpp
        src/partition.cpp
        src/sample.cpp
        src/top_down.cpp
        src/blockmodel/blockmodel.cpp
        src/blockmodel/blockmodel_triplet.cpp
        src/blockmodel/sparse/dict_matrix.cpp
        src/blockmodel/sparse/dict_transpose_matrix.cpp
        src/distributed/dist_block_merge.cpp
        src/distributed/dist_blockmodel_triplet.cpp
        src/distributed/dist_common.cpp
        src/distributed/dist_finetune.cpp
        src/distributed/dist_sbp.cpp
        src/distributed/two_hop_blockmodel.cpp
        #    blockmodel/sparse/dist_dict_matrix.cpp
        src/fastlog.cpp)
target_include_directories(SBP PUBLIC ${INCLUDE_DIRS})

add_executable(DistributedSBP src/main.cpp include/fastlog.hpp)
target_include_directories(DistributedSBP PUBLIC ${INCLUDE_DIRS})

add_executable(Influence src/influence.cpp)
target_include_directories(Influence PUBLIC ${INCLUDE_DIRS})

if (DEBUG)
    #    To check for heap overflows and such, use -fsanitize=address
    #    To run with valgrind, do NOT use -fsanitize=address
    target_compile_options(SBP PUBLIC -Wall -no-pie -fopenmp -g3 -fsanitize=address)
    target_link_libraries(SBP PUBLIC hungarian absl::flat_hash_map -fopenmp stdc++fs -g3 -fsanitize=address tbb)
    target_link_libraries(DistributedSBP PUBLIC SBP absl::flat_hash_map -g3 -fsanitize=address tbb)
    target_link_libraries(Influence PUBLIC SBP absl::flat_hash_map -g3 -fsanitize=address tbb)
    #    target_compile_options(SBP PUBLIC -Wall -no-pie -fopenmp -g3)
    #    target_link_libraries(SBP PUBLIC hungarian -fopenmp stdc++fs -g3)
    #    target_link_libraries(DistributedSBP PUBLIC SBP -g3)
    #    target_link_libraries(Influence PUBLIC SBP -g3)
else (DEBUG)
    #    target_compile_options(DistributedSBP PUBLIC -Wall -O3 -no-pie -fopenmp)
    #    target_link_libraries(DistributedSBP PUBLIC SBP hungarian -fopenmp stdc++fs)
    target_compile_options(SBP PUBLIC -Wall -O3 -no-pie -fopenmp)
    target_link_libraries(SBP PUBLIC hungarian absl::hash absl::flat_hash_map -fopenmp stdc++fs tbb)
    target_link_libraries(DistributedSBP PUBLIC SBP absl::hash absl::flat_hash_map tbb)
    target_link_libraries(Influence PUBLIC SBP absl::hash absl::flat_hash_map tbb)
endif (DEBUG)

# Google Test Suite
enable_testing()
add_subdirectory(extern/googletest)
add_executable(Test
        test/blockmodel_test.cpp
        test/block_merge_test.cpp
        test/common_test.cpp
        test/dict_matrix_test.cpp
        test/dict_transpose_matrix_test.cpp
        test/entropy_test.cpp
        test/evaluate_test.cpp
        test/finetune_test.cpp
        test/sample_test.cpp
        test/toy_example.hpp)
target_include_directories(Test PUBLIC ${INCLUDE_DIRS})
include(GoogleTest)
target_link_libraries(Test SBP gtest_main hungarian absl::hash absl::flat_hash_map -fopenmp stdc++fs tbb)
gtest_discover_tests(Test)
