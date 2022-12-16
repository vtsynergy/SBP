/**
 * The stochastic block blockmodeling module.
 */
#ifndef SBP_SBP_HPP
#define SBP_SBP_HPP

#include <omp.h>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "graph.hpp"

namespace sbp {

/// The total amount of time spent community detection, to be dynamically updated during execution.
extern double total_time;

/// Stores intermediate information for later printing.
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

std::vector<Intermediate> get_intermediates();

/// Performs community detection on the provided graph, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks = 0);

} // namespace sbp

#endif // SBP_SBP_HPP
