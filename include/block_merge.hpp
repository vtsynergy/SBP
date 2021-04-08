/**
 * The block merge phase of the stochastic block blockmodeling module.
 */
#ifndef CPPSBP_BLOCK_MERGE_HPP
#define CPPSBP_BLOCK_MERGE_HPP

#include <limits>
#include <numeric>
#include <random>

// #include <omp.h>
#include "args.hpp"
#include "common.hpp"
#include "blockmodel/blockmodel.hpp"
// #include "blockmodel/sparse/boost_mapped_matrix.hpp"
#include "blockmodel/sparse/csparse_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "utils.hpp"

namespace block_merge {

static const int NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    int proposed_block;
    double delta_entropy;
} ProposalEvaluation;

/// Performs the block merges with the highest change in entropy/MDL, recalculating change in entropy before each
/// merge to account for dependencies between merges. This function modified the blockmodel.
void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                    const std::vector<int> &best_merge_for_each_block);

/// Computes the change in entropy under a proposed block merge using sparse intermediate structures
double compute_delta_entropy_sparse(int current_block, int proposal, Blockmodel &blockmodel,
                                    SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the new edge counts for the affected blocks (communities) 
/// under a proposed block merge
EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks);

/// Fills the new edge counts for the affected blocks (communities) under a proposed block merge.
/// Results are stored as sparse vectors (unordered_maps)
void edge_count_updates_sparse(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates);

/// Merges entire blocks (communities) in blockmodel together
Blockmodel &merge_blocks(Blockmodel &blockmodel, const NeighborList &out_neighbors, Args &args);

/// Proposes a merge for current_block based on the current blockmodel state
ProposalEvaluation propose_merge(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel);

/// Proposes a merge for current_block based on the current blockmodel state, using sparse intermediate structures
ProposalEvaluation propose_merge_sparse(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel,
                                        std::unordered_map<int, bool> &past_proposals);

/// Computes the change in entropy under a proposed block merge
double compute_delta_entropy(int current_block, int proposal, Blockmodel &blockmodel, EdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees);

} // block_merge

#endif // CPPSBP_BLOCK_MERGE_HPP
