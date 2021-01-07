
#include <iostream>
#include <string>

#include "args.hpp"
#include "evaluate.hpp"
#include "graph.hpp"
#include "sbp.hpp"
#include "blockmodel/blockmodel.hpp"


int main(int argc, char* argv[]) {
    Args args(argc, argv);
    std::cout << "Parsed out the arguments" << std::endl;
    Graph graph = Graph::load(args);
    Blockmodel blockmodel = sbp::stochastic_block_blockmodel(graph, args);
    // TODO: make sure evaluate_blockmodel doesn't crash on larger graphs
    evaluate::evaluate_blockmodel(graph, blockmodel);
}
