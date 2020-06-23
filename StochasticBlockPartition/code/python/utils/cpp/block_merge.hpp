/**
 * The block merge phase of the stochastic block partitioning module.
 */
#ifndef CPPSBP_BLOCK_MERGE_HPP
#define CPPSBP_BLOCK_MERGE_HPP

#include <numeric>
#include <random>

#include <omp.h>
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>

#include "common.hpp"
#include "partition/partition.hpp"
#include "partition/sparse/boost_mapped_matrix.hpp"
#include "util/util.hpp"

// namespace py = pybind11;

namespace block_merge {

static const int NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    int proposed_block;
    double delta_entropy;
} ProposalEvaluation;

/*******************
 * BLOCK MERGE
 ******************/
Partition &merge_blocks(Partition &partition, std::vector<Matrix2Column> &out_neighbors);
ProposalEvaluation propose_merge(int current_block, Partition &partition, Vector &block_partition);
// common::ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
//                                         Vector &block_partition, Partition &partition);
EdgeCountUpdates edge_count_updates(BoostMappedMatrix &blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks);
double compute_delta_entropy(int current_block, int proposal, Partition &partition, EdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees);
// int propose_random_block(int current_block, int num_blocks);
// int choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights);
// int choose_neighbor(Eigen::SparseVector<double> &multinomial_distribution);

} // block_merge

#endif // CPPSBP_BLOCK_MERGE_HPP
