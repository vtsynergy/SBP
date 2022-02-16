
#include <chrono>
#include <execinfo.h>
#include <iostream>
#include <mpi.h>
#include <signal.h>
#include <string>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "evaluate.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
#include "sbp.hpp"


// void handler(int sig) {
//     void *array[100];
//     size_t size;
//     size = backtrace(array, 100);
//     fprintf(stderr, "Error: signal %d:\n", sig);
//     backtrace_symbols_fd(array, size, STDERR_FILENO);
//     exit(-1);
// }

MPI_t mpi;
Args args;

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // int rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.num_processes);
    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    args = Args(argc, argv);
    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load(args);

    if (mpi.num_processes > 1) {
        MPI_Barrier(MPI_COMM_WORLD);  // keep start - end as close as possible for all processes
        double start = MPI_Wtime();
        Blockmodel blockmodel = sbp::dist::stochastic_block_partition(graph, args);
        double end = MPI_Wtime();
        if (mpi.rank == 0) {
            evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
            std::cout << "Final F1 score = " << result.f1_score << std::endl;
            std::cout << "Community detection runtime = " << end - start << "s" << std::endl;
        }
        // double avg_f1;
        // Graph partition = partition::partition(graph, mpi.rank, mpi.num_processes, args);
        // Blockmodel partial_blockmodel = sbp::stochastic_block_partition(partition, args);
        // double f1 = evaluate::evaluate_blockmodel(partition, partial_blockmodel);
        // MPI_Reduce(&f1, &avg_f1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // if (mpi.rank == 0) {
        //     avg_f1 /= mpi.num_processes;
        //     std::cout << "Average F1 Score across " << mpi.num_processes << " partitions = " << avg_f1 << std::endl;
        // }
    } else {
        auto start = std::chrono::steady_clock::now();
        Blockmodel blockmodel = sbp::stochastic_block_partition(graph, args);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> runtime = end - start;
        evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
        std::cout << "Final F1 score = " << result.f1_score << std::endl;
        std::cout << "Community detection runtime = " << runtime.count() << "s" << std::endl;
    }

    MPI_Finalize();
}
