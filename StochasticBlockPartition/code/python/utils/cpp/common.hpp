/**
 * Structs and functions common to both the block merge and finetune phases.
 */

#ifndef CPPSBP_COMMON_HPP
#define CPPSBP_COMMON_HPP

#include <vector>

#include <Eigen/Core>

#include "partition/partition.hpp"
#include "partition/sparse/csparse_matrix.hpp"
#include "util/util.hpp"

namespace common {

static std::random_device seeder;
static std::default_random_engine generator(seeder());

typedef struct new_block_degrees_t {
    Vector block_degrees_out;
    Vector block_degrees_in;
    Vector block_degrees;
} NewBlockDegrees;

typedef struct proposal_and_edge_counts_t {
    int proposal;
    int num_out_neighbor_edges;
    int num_in_neighbor_edges;
    int num_neighbor_edges;
} ProposalAndEdgeCounts;

int choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights);
int choose_neighbor(Eigen::SparseVector<double> &multinomial_distribution);
NewBlockDegrees compute_new_block_degrees(int current_block, Partition &partition, common::ProposalAndEdgeCounts &proposal);
double delta_entropy_temp(Vector &row_or_col, Vector &block_degrees, int degree);
Vector exclude_indices(Vector &in, int index1, int index2);
Vector index_nonzero(Vector &values, Vector &indices);
Vector nonzeros(Vector &in);
ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                        Vector &block_partition, Partition &partition, bool block_merge = false);
int propose_random_block(int current_block, int num_blocks);

} // namespace common

#endif // CPPSBP_COMMON_HPP
