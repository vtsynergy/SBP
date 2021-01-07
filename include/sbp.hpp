/**
 * The stochastic block blockmodeling module.
 */
#ifndef SBP_SBP_HPP
#define SBP_SBP_HPP

#include <omp.h>

#include "args.hpp"
#include "block_merge.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "blockmodel/blockmodel_triplet.hpp"

namespace sbp {

/// Performs community detection on the provided graph, using the stochastic block blockmodeling algorithm
Blockmodel stochastic_block_blockmodel(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks = 0);

} // namespace sbp

#endif // SBP_SBP_HPP
