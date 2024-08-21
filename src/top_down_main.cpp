
#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nans or infs
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
//#include "sample.hpp"
#include "sbp.hpp"
#include "top_down.hpp"
#include "utils.hpp"

double sample_time = 0.0;
double sample_extend_time = 0.0;
double finetune_time = 0.0;

MPI_t mpi;
Args args;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime) {
    std::vector<sbp::intermediate> intermediate_results = sbp::get_intermediates();
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
             << "sequential_mcmc_time, parallel_mcmc_time, vertex_move_time, mcmc_moves, total_num_islands, "
             << "block_merge_time, block_merge_loop_time, blockmodel_build_time, finetune_time, "
             << "sort_time, load_balancing_time, access_time, update_assignmnet, total_time" << std::endl;
    }
    for (const sbp::intermediate &temp : intermediate_results) {
        file << args.tag << ", " << graph.num_vertices() << ", " << graph.num_edges() << ", " << args.overlap << ", "
             << args.blocksizevar << ", " << args.undirected << ", " << args.algorithm << ", " << temp.iteration << ", "
             << temp.mdl << ", " << temp.normalized_mdl_v1 << ", " << args.samplesize << ", "
             << temp.modularity << ", " << eval.f1_score << ", " << eval.nmi << ", " << eval.true_mdl << ", "
             << entropy::normalize_mdl_v1(eval.true_mdl, graph) << ", "
             << args.samplingalg << ", " << runtime << ", " << sample_time << ", " << sample_extend_time << ", "
             << finetune_time << ", " << temp.mcmc_iterations << ", " << temp.mcmc_time << ", "
             << temp.mcmc_sequential_time << ", " << temp.mcmc_parallel_time << ", "
             << temp.mcmc_vertex_move_time << ", " << temp.mcmc_moves << ", " << sbp::total_num_islands << ", "
             << temp.block_merge_time << ", " << temp.block_merge_loop_time << ", "
             << temp.blockmodel_build_time << ", " << temp.finetune_time << ", " << temp.sort_time << ", "
             << temp.load_balancing_time << ", " << temp.access_time << ", " << temp.update_assignment << ", "
             << temp.total_time << std::endl;
    }
    file.close();
}

void evaluate_partition(Graph &graph, Blockmodel &blockmodel, double runtime) {
    if (mpi.rank != 0) return;
    utils::write_json(blockmodel.block_assignment(), blockmodel.getOverall_entropy(), finetune::MCMC_moves,
                      finetune::MCMC_iterations, runtime);
    if (!args.evaluate) return;
    evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
    std::cout << "Final F1 score = " << result.f1_score << std::endl;
    std::cout << "Community detection runtime = " << runtime << "s" << std::endl;
    write_results(graph, result, runtime);
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
    if (args.mix)
        partition.blockmodel = top_down::run_mix(partition.graph);
    else
        partition.blockmodel = top_down::run(partition.graph);
    sbp::add_intermediate(-1, partition.graph, -1, partition.blockmodel.getOverall_entropy());
    // evaluate
    double end = MPI_Wtime();
    evaluate_partition(partition.graph, partition.blockmodel, end - start);

    MPI_Finalize();
}
