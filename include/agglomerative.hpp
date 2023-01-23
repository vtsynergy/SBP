//
// Created by Frank on 1/6/2023.
//

#ifndef AGGLOMERATIVESBP_AGGLOMERATIVE_HPP
#define AGGLOMERATIVESBP_AGGLOMERATIVE_HPP

#include "args.hpp"
#include "blockmodel.hpp"
#include "graph.hpp"

namespace agglomerative {

/// Stores intermediate information for later printing.
// TODO: change this to display only agglomerative metrics
struct Intermediate {
    float iteration;
    double mdl;
    double normalized_mdl_v1;
    double modularity;
    int mcmc_iterations;
    double mcmc_time;
    double mcmc_sequential_time;
    double mcmc_parallel_time;
    double mcmc_vertex_move_time;
    uint mcmc_moves;
    double block_merge_time;
    double block_merge_loop_time;
    double blockmodel_build_time;
    double blockmodel_first_build_time;
    double sort_time;
    double access_time;
    double update_assignment;
    double total_time;
};

/// Adds intermediate results to be later saved in a CSV file.
void add_intermediate(float iteration, const Graph &graph, double modularity, double mdl);

Blockmodel agglomerate(Blockmodel &blockmodel, const Graph &graph);

Blockmodel move_vertices(Blockmodel &blockmodel, const Graph &graph);

/// Runs the agglomerative SBP algorithm on `graph` with the given `args`. After each iteration, communities are
/// collapsed into supernodes.
Blockmodel run(Graph &graph, Args &args);

} // namespace agglomerative

#endif //AGGLOMERATIVESBP_AGGLOMERATIVE_HPP
