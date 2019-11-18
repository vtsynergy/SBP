/**
 * The stochastic block partitioning module.
 */
#ifndef CPPSBP_SBP_HPP
#define CPPSBP_SBP_HPP

#include <numeric>
#include <random>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "partition/partition.hpp"
#include "partition/sparse/boost_mapped_matrix.hpp"
#include "util/util.hpp"

namespace py = pybind11;

namespace sbp {

static std::random_device seeder;
static std::default_random_engine generator(seeder());

typedef struct proposal_evaluation_t {
    int proposed_block;
    double delta_entropy;
} ProposalEvaluation;

typedef struct proposal_and_edge_counts_t {
    int proposal;
    int num_out_neighbor_edges;
    int num_in_neighbor_edges;
    int num_neighbor_edges;
} ProposalAndEdgeCounts;

typedef struct block_merge_edge_count_updates_t {
    // Vector block_row;  All zeros, so no point in storing
    Vector proposal_row;
    // Vector block_col;  All zeros, so no point in storing
    Vector proposal_col;
} BlockMergeEdgeCountUpdates;

typedef struct new_block_degrees_t {
    Vector block_degrees_out;
    Vector block_degrees_in;
    Vector block_degrees;
} NewBlockDegrees;

/*******************
 * BLOCK MERGE
 ******************/
Partition &merge_blocks(Partition &partition, int num_agg_proposals_per_block,
                        std::vector<Matrix2Column> &out_neighbors);
ProposalEvaluation propose_merge(int current_block, Partition &partition, Vector &block_partition);
ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                        Vector &block_partition, Partition &partition);
BlockMergeEdgeCountUpdates block_merge_edge_count_updates(BoostMappedMatrix &blockmodel, int current_block,
                                                          int proposed_block, EdgeWeights &out_blocks,
                                                          EdgeWeights &in_blocks);
NewBlockDegrees compute_new_block_degrees(int current_block, Partition &partition, ProposalAndEdgeCounts &proposal);
double compute_delta_entropy(int current_block, int proposal, Partition &partition, BlockMergeEdgeCountUpdates &updates,
                             NewBlockDegrees &block_degrees);
int propose_random_block(int current_block, int num_blocks);
int choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights);
int choose_neighbor(Eigen::SparseVector<double> &multinomial_distribution);

/*******************
 * FINE-TUNE
 ******************/
} // namespace sbp

#endif // CPPSBP_SBP_HPP
