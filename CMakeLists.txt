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
add_subdirectory(extern/abseil-cpp)

set(INCLUDE_DIRS
    include
    include/blockmodel
    include/blockmodel/sparse
    extern/tclap-1.2.2/include
    patches
    extern/hungarian
    extern/abseil-cpp
        )

# add the executable
add_library(SBP STATIC
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
#    blockmodel/sparse/dist_dict_matrix.cpp
        )
target_include_directories(SBP PUBLIC ${INCLUDE_DIRS})

add_executable(DistributedSBP main.cpp)
target_include_directories(DistributedSBP PUBLIC ${INCLUDE_DIRS})

add_executable(Influence influence.cpp)
target_include_directories(Influence PUBLIC ${INCLUDE_DIRS})

if(DEBUG)
#    To check for heap overflows and such, use -fsanitize=address
#    To run with valgrind, do NOT use -fsanitize=address
    target_compile_options(SBP PUBLIC -Wall -no-pie -fopenmp -g3 -fsanitize=address)
    target_link_libraries(SBP PUBLIC hungarian absl::flat_hash_map -fopenmp stdc++fs -g3 -fsanitize=address)
    target_link_libraries(DistributedSBP PUBLIC SBP absl::flat_hash_map -g3 -fsanitize=address)
    target_link_libraries(Influence PUBLIC SBP absl::flat_hash_map -g3 -fsanitize=address)
#    target_compile_options(SBP PUBLIC -Wall -no-pie -fopenmp -g3)
#    target_link_libraries(SBP PUBLIC hungarian -fopenmp stdc++fs -g3)
#    target_link_libraries(DistributedSBP PUBLIC SBP -g3)
#    target_link_libraries(Influence PUBLIC SBP -g3)
else(DEBUG)
#    target_compile_options(DistributedSBP PUBLIC -Wall -O3 -no-pie -fopenmp)
#    target_link_libraries(DistributedSBP PUBLIC SBP hungarian -fopenmp stdc++fs)
    target_compile_options(SBP PUBLIC -Wall -O3 -no-pie -fopenmp)
    target_link_libraries(SBP PUBLIC hungarian absl::hash absl::flat_hash_map -fopenmp stdc++fs)
    target_link_libraries(DistributedSBP PUBLIC SBP absl::hash absl::flat_hash_map)
    target_link_libraries(Influence PUBLIC SBP absl::hash absl::flat_hash_map)
endif(DEBUG)

# Google Test Suite
enable_testing()
add_subdirectory(extern/googletest)
add_executable(Test
        test/blockmodel_test.cpp
        test/block_merge_test.cpp
        test/common_test.cpp
        test/finetune_test.cpp
        test/toy_example.hpp)
target_include_directories(Test PUBLIC ${INCLUDE_DIRS})
include(GoogleTest)
target_link_libraries(Test SBP gtest_main hungarian absl::hash absl::flat_hash_map -fopenmp stdc++fs)
gtest_discover_tests(Test)
