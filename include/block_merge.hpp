/**
 * The block merge phase of the stochastic block blockmodeling module.
 */
#ifndef CPPSBP_BLOCK_MERGE_HPP
#define CPPSBP_BLOCK_MERGE_HPP

#include <numeric>
#include <random>

// #include <omp.h>

#include "common.hpp"
#include "blockmodel/blockmodel.hpp"
// #include "blockmodel/sparse/boost_mapped_matrix.hpp"
#include "blockmodel/sparse/dict_transpose_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "utils.hpp"

namespace block_merge {

static const int NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    int proposed_block;
    double delta_entropy;
} ProposalEvaluation;

/// Merges entire blocks (communities) in blockmodel together
Blockmodel &merge_blocks(Blockmodel &blockmodel, NeighborList &out_neighbors);

/// Proposes a merge for current_block based on the current blockmodel state
ProposalEvaluation propose_merge(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel);

/// Proposes a merge for current_block based on the current blockmodel state, using sparse intermediate structures
ProposalEvaluation propose_merge_sparse(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel);

/// Computes the new edge counts for the affected blocks (communities) 
/// under a proposed block merge
EdgeCountUpdates edge_count_updates(DictTransposeMatrix &blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks);

/// Fills the new edge counts for the affected blocks (communities) under a proposed block merge.
/// Results are stored as sparse vectors (unordered_maps)
void edge_count_updates_sparse(DictTransposeMatrix &blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates);

/// Computes the change in entropy under a proposed block merge
double compute_delta_entropy(int current_block, int proposal, Blockmodel &blockmodel, EdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees);

/// Computes the change in entropy under a proposed block merge using sparse intermediate structures
double compute_delta_entropy_sparse(int current_block, int proposal, Blockmodel &blockmodel,
                                    SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

} // block_merge

#endif // CPPSBP_BLOCK_MERGE_HPP
