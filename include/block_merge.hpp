/**
 * The block merge phase of the stochastic block partitioning module.
 */
#ifndef CPPSBP_BLOCK_MERGE_HPP
#define CPPSBP_BLOCK_MERGE_HPP

#include <numeric>
#include <random>

// #include <omp.h>

#include "common.hpp"
#include "partition/partition.hpp"
// #include "partition/sparse/boost_mapped_matrix.hpp"
#include "partition/sparse/dict_matrix.hpp"
#include "partition/sparse/typedefs.hpp"
#include "utils.hpp"

namespace block_merge {

static const int NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    int proposed_block;
    double delta_entropy;
} ProposalEvaluation;

/// Merges entire blocks (communities) in partition together
Partition &merge_blocks(Partition &partition, NeighborList &out_neighbors);

/// Proposes a merge for current_block based on the current partition state
ProposalEvaluation propose_merge(int current_block, Partition &partition, std::vector<int> &block_partition);

/// Computes the new edge counts for the affected blocks (communities) 
/// under a proposed block merge
EdgeCountUpdates edge_count_updates(DictMatrix &blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks);

/// Computes the change in entropy under a proposed block merge
double compute_delta_entropy(int current_block, int proposal, Partition &partition, EdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees);

} // block_merge

#endif // CPPSBP_BLOCK_MERGE_HPP
