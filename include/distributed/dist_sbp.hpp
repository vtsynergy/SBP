/**
 * The distributed stochastic block blockmodeling module.
 */
#ifndef SBP_DIST_SBP_HPP
#define SBP_DIST_SBP_HPP

#include "args.hpp"
#include "blockmodel.hpp"
#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/two_hop_blockmodel.hpp"
#include "graph.hpp"

namespace sbp::dist {

/// Performs community detection on the provided graph using MPI, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided distributed blockmodels
bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet,
                        int min_num_blocks = 0);

} // namespace sbp::dist

#endif  // SBP_DIST_SBP_HPP