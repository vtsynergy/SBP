
#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nans or infs
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

#include "args.hpp"
#include "block_merge.hpp"
#include "blockmodel/blockmodel.hpp"
#include "distributed/dist_sbp.hpp"
#include "distributed/divide_and_conquer.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
#include "rng.hpp"
#include "sample.hpp"
#include "sbp.hpp"


MPI_t mpi;
Args args;

//const int NUM_VERTICES_TAG = 0;
//const int VERTICES_TAG = 1;
//const int BLOCKS_TAG = 2;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void run(Partition &partition) {
    if (mpi.num_processes > 1) {
//        MPI_Barrier(mpi.comm);  // keep start - end as close as possible for all processes
//        double start = MPI_Wtime();
        partition.blockmodel = sbp::dist::stochastic_block_partition(partition.graph, args);
//        double end = MPI_Wtime();
//        if (mpi.rank == 0)
//        evaluate_partition(partition.graph, partition.blockmodel, end - start);
    } else {
//        auto start = std::chrono::steady_clock::now();
        partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args);
//        auto end = std::chrono::steady_clock::now();
//        evaluate_partition(partition.graph, partition.blockmodel, std::chrono::duration<double>(end - start).count());
    }
}

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // long rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.num_processes);
    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    args = Args(argc, argv);
    rng::init_generators();

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load();
    sample::Sample subgraph = sample::round_robin(graph, mpi.rank, mpi.num_processes);
    Partition partition;
    double start = MPI_Wtime();
    partition.graph = std::move(subgraph.graph);
    // TODO: add stopping at golden ratio
    partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args, true);
    double end_blockmodeling = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << mpi.rank << " took " << end_blockmodeling - start << "s to finish runtime | final B = "
              << partition.blockmodel.getNum_blocks() << std::endl;
//
//    std::vector<std::vector<long>> rank_vertices;
//    std::vector<std::vector<long>> rank_assignment;
//
//    // Compute local partition information
//    int local_num_vertices = subgraph.graph.num_vertices();
//    std::vector<long> local_vertices = utils::constant<long>(subgraph.graph.num_vertices(), -1);
//    std::vector<long> local_assignment = utils::constant<long>(subgraph.graph.num_vertices(), -1);
//    #pragma omp parallel for schedule(dynamic) default(none) \
//            shared(graph, subgraph, partition, local_vertices, local_assignment)
//    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
//        long subgraph_index = subgraph.mapping[vertex];
//        if (subgraph_index < 0) continue;  // vertex not present
//        long assignment = partition.blockmodel.block_assignment(subgraph_index);
//        local_vertices[subgraph_index] = vertex;
//        local_assignment[subgraph_index] = assignment;
//    }
//    rank_vertices.push_back(local_vertices);
//    rank_assignment.push_back(local_assignment);
//
//    // For some reason program hangs using MPI_Send and MPI_Recv. Buffer full or something like that?
//    MPI_Barrier(MPI_COMM_WORLD);
//    std::cout << mpi.rank << " | Ranks about to start combining partial results (1)" << std::endl;
//    int num_vertices[mpi.num_processes];
////    MPI_Allgather(&local_num_vertices, 1, MPI_INT, &num_vertices, 1, MPI_INT, MPI_COMM_WORLD);
//    int result;
//    result = MPI_Gather(&local_num_vertices, 1, MPI_INT, num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
//    if (result != MPI_SUCCESS) {
//        std::cerr << mpi.rank << " | Something went wrong, could not obtain number of vertices!" << std::endl;
//    }
//    std::cout << mpi.rank << " | All ranks sent their vertex numbers (2)" << std::endl;
//    int offsets[mpi.num_processes];
//    offsets[0] = 0;
//    for (int rank = 1; rank < mpi.num_processes; ++rank) {
//        offsets[rank] = num_vertices[rank - 1] + offsets[rank - 1];
//    }
//    std::vector<long> all_vertices = utils::constant<long>(graph.num_vertices(), -1);
//    std::vector<long> all_assignments = utils::constant<long>(graph.num_vertices(), -1);
//    result = MPI_Gatherv(local_vertices.data(), local_num_vertices, MPI_LONG, all_vertices.data(), num_vertices, offsets, MPI_LONG, 0, MPI_COMM_WORLD);
//    if (result != MPI_SUCCESS) {
//        std::cerr << mpi.rank << " | Something went wrong, could not obtain vertex information!" << std::endl;
//    }
//    std::cout << mpi.rank << " | All ranks have sent their vertex ids (3)" << std::endl;
//    result = MPI_Gatherv(local_assignment.data(), local_num_vertices, MPI_LONG, all_assignments.data(), num_vertices, offsets, MPI_LONG, 0, MPI_COMM_WORLD);
//    if (result != MPI_SUCCESS) {
//        std::cerr << mpi.rank << " | Something went wrong, assignment information!" << std::endl;
//    }
//    std::cout << mpi.rank << " | All ranks have completed communication (4)" << std::endl;
//
//    if (mpi.rank == 0) {
//        // Fill up rank_vertices and rank_assignment
//        for (int rank = 1; rank < mpi.num_processes; ++rank) {
//            int offset = offsets[rank];
//            int num_v = num_vertices[rank];
//            std::vector<long> rank_vertex_list = utils::constant<long>(num_v, -1);
//            std::vector<long> rank_assignment_list = utils::constant<long>(num_v, -1);
//            for (int index = 0; index < num_v; ++index) {
//                rank_vertex_list[index] = all_vertices[index + offset];
//                rank_assignment_list[index] = all_assignments[index + offset];
//            }
//            rank_vertices.push_back(rank_vertex_list);
//            rank_assignment.push_back(rank_assignment_list);
//        }
//        // Start the combination process
//        long offset = 0;
//        std::vector<long> combined_assignment = dnc::combine_partitions(graph, offset, rank_vertices, rank_assignment);
//        std::cout << "Done combining partitions" << std::endl;
//        Blockmodel blockmodel(offset, graph, 0.25, combined_assignment);
//        std::cout << "Done building blockmodel" << std::endl;
//        blockmodel = dnc::finetune_partition(blockmodel, graph);
//        std::cout << "Done finetuning result" << std::endl;
//        // only last iteration result will calculate expensive modularity
//        double modularity = -1;
//        if (args.modularity)
//            modularity = graph.modularity(blockmodel.block_assignment());
//        sbp::add_intermediate(-1, graph, modularity, blockmodel.getOverall_entropy());
//        // Evaluate finetuned assignment
//        double end = MPI_Wtime();
//        dnc::evaluate_partition(graph, blockmodel, end - start);
//    }
    MPI_Finalize();
}
