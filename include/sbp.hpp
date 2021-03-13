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

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks = 0);

/// A hierarchical iteration of the stochastic block partitioning algorithm. Reduce number of blocks in increments,
/// with each increment reducing the number of blocks by a factor of 1.3. After each increment, move vertices
/// around. The version used in the graph-tool package.
Blockmodel hierarchical_iteration(Blockmodel &blockmodel, const Graph &graph, BlockmodelTriplet &blockmodel_triplet,
                                  Args &args);

/// Returns the number of blocks to merge in the next hierarchical inner iteration.
int next_num_blocks_to_merge(Blockmodel &blockmodel, int target_num_blocks);

/// A flat (non-hierarchical) iteration of the stochastic block partitioning algorithm. Reduce number of blocks, then
/// move vertices around. The version used in the Graph Challenge codebase.
Blockmodel flat_iteration(Blockmodel &blockmodel, const Graph &graph, BlockmodelTriplet &blockmodel_triplet,
                          Args &args);

/// Performs community detection on the provided graph, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(const Graph &graph, Args &args);

} // namespace sbp

#endif // SBP_SBP_HPP
