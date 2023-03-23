
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

double sample_time = 0.0;
double sample_extend_time = 0.0;
double finetune_time = 0.0;

//const int NUM_VERTICES_TAG = 0;
//const int VERTICES_TAG = 1;
//const int BLOCKS_TAG = 2;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime) {
    std::vector<sbp::intermediate> intermediate_results;
    intermediate_results = sbp::get_intermediates();
    std::ostringstream filepath_stream;
    filepath_stream << args.csv << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.type << ".csv";
    std::string filepath = filepath_stream.str();
    bool file_exists = fs::exists(filepath);
    std::cout << std::boolalpha <<  "writing results to " << filepath << " exists = " << file_exists << std::endl;
    fs::create_directories(fs::path(filepath_dir));
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if (!file_exists) {
        file << "tag, numvertices, numedges, overlap, blocksizevar, undirected, algorithm, iteration, mdl, "
             << "normalized_mdl_v1, sample_size, modularity, f1_score, nmi, true_mdl, true_mdl_v1, sampling_algorithm, "
             << "runtime, sampling_time, sample_extend_time, finetune_time, mcmc_iterations, mcmc_time, "
             << "sequential_mcmc_time, parallel_mcmc_time, vertex_move_time, mcmc_moves, block_merge_time, "
             << "block_merge_loop_time, blockmodel_build_time, first_blockmodel_build_time, sort_time, "
             << "load_balancing_time, access_time, update_assignmnet, total_time" << std::endl;
    }
    for (const sbp::intermediate &temp : intermediate_results) {
        file << args.tag << ", " << graph.num_vertices() << ", " << graph.num_edges() << ", " << args.overlap << ", "
             << args.blocksizevar << ", " << args.undirected << ", " << args.algorithm << ", " << temp.iteration << ", "
             << temp.mdl << ", " << temp.normalized_mdl_v1 << ", " << args.samplesize << ", "
             << temp.modularity << ", " << eval.f1_score << ", " << eval.nmi << ", " << eval.true_mdl << ", "
             << entropy::normalize_mdl_v1(eval.true_mdl, graph.num_edges()) << ", "
             << args.samplingalg << ", " << runtime << ", " << sample_time << ", " << sample_extend_time << ", "
             << finetune_time << ", " << temp.mcmc_iterations << ", " << temp.mcmc_time << ", "
             << temp.mcmc_sequential_time << ", " << temp.mcmc_parallel_time << ", "
             << temp.mcmc_vertex_move_time << ", " << temp.mcmc_moves << ", " << temp.block_merge_time << ", "
             << temp.block_merge_loop_time << ", " << temp.blockmodel_build_time << ", "
             << temp.blockmodel_first_build_time << ", " << temp.sort_time << ", " << temp.load_balancing_time << ", "
             << temp.access_time << ", " << temp.update_assignment << ", " << temp.total_time << std::endl;
    }
    file.close();
}

void evaluate_partition(Graph &graph, Blockmodel &blockmodel, double runtime) {
    if (mpi.rank != 0) return;
    evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
    std::cout << "Final F1 score = " << result.f1_score << std::endl;
    std::cout << "Community detection runtime = " << runtime << "s" << std::endl;
    write_results(graph, result, runtime);
}

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
    std::cout << "Rank " << mpi.rank << " took " << end_blockmodeling - start << "s to finish runtime | final B = "
              << partition.blockmodel.getNum_blocks() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<std::vector<long>> rank_vertices;
    std::vector<std::vector<long>> rank_assignment;

    // Compute local partition information
    int local_num_vertices = subgraph.graph.num_vertices();
    std::vector<long> local_vertices = utils::constant<long>(subgraph.graph.num_vertices(), -1);
    std::vector<long> local_assignment = utils::constant<long>(subgraph.graph.num_vertices(), -1);
    #pragma omp parallel for schedule(dynamic) default(none) \
            shared(graph, subgraph, partition, local_vertices, local_assignment)
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long subgraph_index = subgraph.mapping[vertex];
        if (subgraph_index < 0) continue;  // vertex not present
        long assignment = partition.blockmodel.block_assignment(subgraph_index);
        local_vertices[subgraph_index] = vertex;
        local_assignment[subgraph_index] = assignment;
    }

    if (mpi.rank == 0) {
        rank_vertices.push_back(local_vertices);
        rank_assignment.push_back(local_assignment);
        // Receive data from all processes
        for (int rank = 1; rank < mpi.num_processes; ++rank) {
            dnc::receive_partition(rank, rank_vertices, rank_assignment);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        // The sender
        // Send partition information to root
        MPI_Send(&local_num_vertices, 1, MPI_INT, 0, NUM_VERTICES_TAG, MPI_COMM_WORLD);
        MPI_Send(local_vertices.data(), local_num_vertices, MPI_LONG, 0, VERTICES_TAG, MPI_COMM_WORLD);
        MPI_Send(local_assignment.data(), local_num_vertices, MPI_LONG, 0, BLOCKS_TAG, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (mpi.rank == 0) {
        long offset = 0;
        std::vector<long> combined_assignment = dnc::combine_partitions(graph, offset, rank_vertices, rank_assignment);
        Blockmodel blockmodel(offset, graph, 0.25, combined_assignment);
        blockmodel = dnc::finetune_partition(blockmodel, graph);
        // only last iteration result will calculate expensive modularity
        double modularity = -1;
        if (args.modularity)
            modularity = graph.modularity(blockmodel.block_assignment());
        sbp::add_intermediate(-1, graph, modularity, blockmodel.getOverall_entropy());
        // Evaluate finetuned assignment
        double end = MPI_Wtime();
        evaluate_partition(graph, blockmodel, end - start);
    }
    MPI_Finalize();
}
