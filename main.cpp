
#include <iostream>
#include <string>

#include "args.hpp"
#include "evaluate.hpp"
#include "graph.hpp"
#include "sbp.hpp"
#include "partition/partition.hpp"


int main(int argc, char* argv[]) {
    Args args(argc, argv);
    std::cout << "Parsed out the arguments" << std::endl;
    Graph graph = Graph::load(args);
    Partition blockmodel = sbp::stochastic_block_partition(graph, args);
    // TODO: make sure evaluate_partition doesn't crash on larger graphs
    evaluate::evaluate_partition(graph, blockmodel);
}
