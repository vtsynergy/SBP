cmake_minimum_required(VERSION 3.10)

# set compilers
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

# set the project name
project(DistributedSBP VERSION 2021.08.29)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(hungarian)

# add the executable
add_executable(DistributedSBP main.cpp)
target_include_directories(DistributedSBP PUBLIC
    include
    include/blockmodel
    include/blockmodel/sparse
    tclap-1.2.2/include
    patches
    hungarian)

target_sources(DistributedSBP PUBLIC
    # main.cpp
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

target_compile_options(DistributedSBP PUBLIC -Wall -O3 -no-pie -fopenmp)  #  -lstdc++fs)

target_link_libraries(DistributedSBP PUBLIC hungarian -fopenmp stdc++fs)