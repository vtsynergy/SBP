
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

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void run(Partition &partition) {
    if (mpi.num_processes > 1) {
//        MPI_Barrier(MPI_COMM_WORLD);  // keep start - end as close as possible for all processes
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

int GlobalRank;
int TotalRanks;

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // long rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &TotalRanks);
    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    // Heracles
    // HyDeS/HaDeS (Hierarchical Distributed Sbp)
    // STRiDeS (Stratified Distributed Sbp)
    // HiERArChical distributed Stochastic block partitioning

    // New argument : subgraphs
    // Initially, there are `subgraphs` groups | subgraphs <= mpi.num_processes
    // Rank within subgraph = rank % subgraphs
    // Need to pass communicator to distributed SBP functions for communication purposes
    // Need to ensure that every rank within a subgraph gets the same subgraph

    args = Args(argc, argv);

    int ranks_in_color = ceil(double(TotalRanks) / double(args.subgraphs));
    int color = GlobalRank / ranks_in_color;
//    if (args.subgraphs == 1) color = 0;
    MPI_Comm_split(MPI_COMM_WORLD, color, GlobalRank % args.subgraphs, &mpi.comm);
    rng::init_generators();

    MPI_Comm_rank(mpi.comm, &mpi.rank);
    MPI_Comm_size(mpi.comm, &mpi.num_processes);

    std::cout << "G" << GlobalRank << " L" << mpi.rank << " (" << color << ") | can see " << mpi.num_processes << " processes." << std::endl;

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << TotalRanks << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load();
    sample::Sample subgraph = sample::round_robin(graph, color, args.subgraphs);

    std::cout << "G" << GlobalRank << " L" << mpi.rank << " (" << color << ") | can see " << subgraph.graph.num_vertices() << " V and E = " << subgraph.graph.num_edges() << std::endl;

    std::cout << "G" << GlobalRank << " L" << mpi.rank << " (" << color << ") | can see " << mpi.num_processes << " processes and is processing G with size: " << subgraph.graph.num_vertices() << std::endl;

    Partition partition;
    double start = MPI_Wtime();
    partition.graph = std::move(subgraph.graph);
    // TODO: add stopping at golden ratio
//    partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args, true);
    partition.blockmodel = sbp::dist::stochastic_block_partition(partition.graph, args, true);
    double end_blockmodeling = MPI_Wtime();
    std::cout << "Rank " << mpi.rank << " took " << end_blockmodeling - start << "s to finish initial partitioning | final B = "
              << partition.blockmodel.getNum_blocks() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<std::vector<long>> rank_vertices;
    std::vector<std::vector<long>> rank_assignment;
    std::vector<long> local_vertices, local_assignment;
    dnc::translate_local_partition(local_vertices, local_assignment, subgraph, graph.num_vertices(),
                                   partition.blockmodel.block_assignment());
    int local_num_vertices = subgraph.graph.num_vertices();
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
//
//    if (GlobalRank == 0) {
//        rank_vertices.push_back(local_vertices);
//        rank_assignment.push_back(local_assignment);
//        // Receive data from all processes
//        for (int rank = ranks_in_color; rank < TotalRanks; rank += ranks_in_color) {
//            dnc::receive_partition(rank, rank_vertices, rank_assignment);
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    } else {
//        if (mpi.rank == 0) {  // Only the first rank in each color should send the messages
//            // The sender
//            // Send partition information to root
//            std::cout << "rank " << GlobalRank << " sending info to root..." << std::endl;
//            MPI_Send(&local_num_vertices, 1, MPI_INT, 0, NUM_VERTICES_TAG, MPI_COMM_WORLD);
//            MPI_Send(local_vertices.data(), local_num_vertices, MPI_LONG, 0, VERTICES_TAG, MPI_COMM_WORLD);
//            MPI_Send(local_assignment.data(), local_num_vertices, MPI_LONG, 0, BLOCKS_TAG, MPI_COMM_WORLD);
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
//
//    if (GlobalRank == 0) {
//        long offset = 0;
//        std::vector<long> combined_assignment = dnc::combine_partitions(graph, offset, rank_vertices, rank_assignment);
//        Blockmodel blockmodel(offset, graph, 0.25, combined_assignment);
//        // Make this distributed?
//        blockmodel = dnc::finetune_partition(blockmodel, graph);
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
