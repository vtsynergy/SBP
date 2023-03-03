
#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nans or infs
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "distributed/dist_sbp.hpp"
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

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime) {
    std::vector<sbp::intermediate> intermediate_results;
    if (mpi.num_processes > 1) {
        intermediate_results = sbp::dist::get_intermediates();
    } else {
        intermediate_results = sbp::get_intermediates();
    }
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
    sample::Sample detached;
    Partition partition;
    double start = MPI_Wtime();
    if (args.detach) {
        detached = sample::detach(graph);
        partition.graph = std::move(detached.graph);
        std::cout << "detached num vertices: " << partition.graph.num_vertices() << " E: "
                  << partition.graph.num_edges() << std::endl;
    } else {
        partition.graph = std::move(graph);
    }
    if (args.samplesize <= 0.0) {
        std::cerr << "ERROR " << "Sample size of " << args.samplesize << " is too low. Must be greater than 0.0" << std::endl;
        exit(-5);
    } else if (args.samplesize < 1.0) {
        double sample_start_t = MPI_Wtime();
        std::cout << "Running sampling with size: " << args.samplesize << std::endl;
//        sample::Sample s = sample::max_degree(partition.graph);
        sample::Sample s = sample::sample(partition.graph);
        Partition sample_partition;
        sample_partition.graph = std::move(s.graph);  // s.graph may be empty now
        // add timer
        double sample_end_t = MPI_Wtime();
        sample_time = sample_end_t - sample_start_t;
        run(sample_partition);
        double extend_start_t = MPI_Wtime();
        s.graph = std::move(sample_partition.graph);  // refill s.graph
        // extend sample to full graph
        partition.blockmodel = sample::extend(partition.graph, sample_partition.blockmodel, s);
        // fine-tune full graph
        double finetune_start_t = MPI_Wtime();
        partition.blockmodel = finetune::finetune_assignment(partition.blockmodel, partition.graph);
        double finetune_end_t = MPI_Wtime();
        sample_extend_time = finetune_start_t - extend_start_t;
        finetune_time = finetune_end_t - finetune_start_t;
    } else {
        std::cout << "Running without sampling." << std::endl;
        run(partition);
    }
    if (args.detach) {
        std::cout << "Reattaching island and 1-degree vertices" << std::endl;
        partition.blockmodel = sample::reattach(graph, partition.blockmodel, detached);
    } else {
        graph = std::move(partition.graph);
    }
    // evaluate
    double end = MPI_Wtime();
    evaluate_partition(graph, partition.blockmodel, end - start);

    MPI_Finalize();
}
