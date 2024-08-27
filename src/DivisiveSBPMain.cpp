
#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nans or infs
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "distributed/dist_divisive_sbp.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "globals.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
//#include "sample.hpp"
#include "divisive_sbp.hpp"
#include "utils.hpp"

//double sample_time = 0.0;
//double sample_extend_time = 0.0;
//double finetune_time = 0.0;

//MPI_t mpi;
//Args args;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void evaluate_partition(Graph &graph, Blockmodel &blockmodel, double runtime) {
    if (mpi.rank != 0) return;
    utils::write_json(blockmodel.block_assignment(), blockmodel.getOverall_entropy(), timers::MCMC_moves,
                      timers::MCMC_iterations, runtime);
    std::cout << std::boolalpha << "args.evaluate: " << args.evaluate << std::endl;
    if (!args.evaluate) return;
    std::cout << "evaluatin' ..." << std::endl;
    evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
    std::cout << "Final F1 score = " << result.f1_score << std::endl;
    std::cout << "Community detection runtime = " << runtime << "s" << std::endl;
    evaluate::write_results(graph, result, runtime);
}

void run(Partition &partition) {
    timers::total_num_islands = partition.graph.num_islands();
    if (mpi.num_processes > 1) {
        std::cout << "Distributed Divisive SBP not fully implemented yet!" << std::endl;
        partition.blockmodel = divisive::dist::run(partition.graph);
    } else {
        partition.blockmodel = divisive::run(partition.graph);
    }
    double mdl = partition.blockmodel.getOverall_entropy();
    utils::save_partial_profile(-1, -1, mdl, entropy::normalize_mdl_v1(mdl, partition.graph));
}

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // int rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.num_processes);
    args = Args(argc, argv);
    rng::init_generators();  // TO-DO: automagically init generators. Initialized = false?
    DIVISIVE_SBP = true;
    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Partition partition;
    partition.graph = Graph::load();
    double start = MPI_Wtime();
    run(partition);
    // evaluate
    double end = MPI_Wtime();
    evaluate_partition(partition.graph, partition.blockmodel, end - start);

    MPI_Finalize();
}
