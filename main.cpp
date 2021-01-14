
#include <execinfo.h>
#include <iostream>
#include <mpi.h>
#include <signal.h>
#include <string>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "evaluate.hpp"
#include "graph.hpp"
#include "partition.hpp"
#include "sbp.hpp"


void handler(int sig) {
    void *array[100];
    size_t size;
    size = backtrace(array, 100);
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(-1);
}


int main(int argc, char* argv[]) {
    signal(SIGABRT, handler);
    int rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    Args args(argc, argv);
    if (rank == 0) {
        std::cout << "Number of processes = " << num_processes << std::endl;
        std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load(args);

    if (num_processes > 1) {
        if (rank == 0) {  // TODO: return this once we figure out the problem
            for (int i = 0; i < num_processes; ++i) {
                Graph partition = partition::partition(graph, i, num_processes, args);
                Blockmodel partial_blockmodel = sbp::stochastic_block_partition(partition, args);
                evaluate::evaluate_blockmodel(partition, partial_blockmodel);
            }
        }
    } else {
        double avg_f1 = 0.0;
        for (int i = 0; i < args.nodes; ++i) {
            Graph partition = partition::partition(graph, i, args.nodes, args);
            Blockmodel partial_blockmodel = sbp::stochastic_block_partition(partition, args);
            avg_f1 += evaluate::evaluate_blockmodel(partition, partial_blockmodel);
        }
        avg_f1 /= (double) args.nodes;
        std::cout << "average f1 score across " << args.nodes << " partitions = " << avg_f1 << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
        // Blockmodel blockmodel = sbp::stochastic_block_partition(graph, args);
        // // TODO: make sure evaluate_blockmodel doesn't crash on larger graphs
        // evaluate::evaluate_blockmodel(graph, blockmodel);
    }

    MPI_Finalize();
}
