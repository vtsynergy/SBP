/**
 * The block merge phase of the stochastic block blockmodeling module.
 */
#ifndef SBP_BLOCK_MERGE_HPP
#define SBP_BLOCK_MERGE_HPP

#include <limits>
#include <numeric>
#include <random>

// #include <omp.h>
#include "args.hpp"
#include "common.hpp"
#include "blockmodel/blockmodel.hpp"
// #include "blockmodel/sparse/boost_mapped_matrix.hpp"
#include "blockmodel/sparse/dict_transpose_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "mpi.h"
#include "mpi_utils.hpp"
#include "partition.hpp"

namespace sbp {

namespace block_merge {

static const int NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    int proposed_block;
    double delta_entropy;
} ProposalEvaluation;

namespace parallel {

/// Performs the block merges with the highest change in entropy/MDL, recalculating change in entropy before each
/// merge to account for dependencies between merges. This function modified the blockmodel.
void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                    const std::vector<int> &best_merge_for_each_block);

/// Merges entire blocks (communities) in blockmodel together
Blockmodel &merge_blocks(Blockmodel &blockmodel, NeighborList &out_neighbors, Args &args);

/// Proposes a merge for current_block based on the current blockmodel state
ProposalEvaluation propose_merge(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel);

/// Proposes a merge for current_block based on the current blockmodel state, using sparse intermediate structures
ProposalEvaluation propose_merge_sparse(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel,
                                        std::unordered_map<int, bool> &past_proposals);

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

} // namespace parallel

namespace naive_distributed {

/// Fills the new edge counts for the affected blocks (communities) under a proposed block merge. Requires communication
/// with other MPI nodes to retrieve/send information. Results are stored as sparse vectors (unordered_maps)
void edge_count_updates_sparse(partition::BlockmodelPartition &partition, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates,
                               utils::mpi::Info &mpi);

/// Merges entire blocks (communities) in blockmodel together in a distributed manner.
partition::BlockmodelPartition &merge_blocks(partition::BlockmodelPartition &blockmodel,
                                             const NeighborList &out_neighbors, utils::mpi::Info &mpi, Args &args);

/// Proposes a merge for current_block based on the current blockmodel state, using sparse intermediate structures.
/// Requires communication with other MPI nodes to retrieve/send information.
ProposalEvaluation propose_merge_sparse(int current_block, partition::BlockmodelPartition &partition,
                                        std::vector<int> &block_assignment,
                                        std::unordered_map<int, bool> &past_proposals, utils::mpi::Info &mpi);

} // namespace naive_distributed

} // namespace block_merge

} // namespace sbp

#endif // SBP_BLOCK_MERGE_HPP
