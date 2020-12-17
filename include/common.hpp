/**
 * Structs and functions common to both the block merge and finetune phases.
 */

#ifndef SBP_COMMON_HPP
#define SBP_COMMON_HPP

#include <random>
#include <vector>

// #include <Eigen/Core>

#include "partition/partition.hpp"
#include "partition/sparse/csparse_matrix.hpp"
#include "partition/sparse/typedefs.hpp"
#include "utils.hpp"

namespace common {

static std::random_device seeder;
static std::default_random_engine generator(seeder());

typedef struct new_block_degrees_t {
    std::vector<int> block_degrees_out;
    std::vector<int> block_degrees_in;
    std::vector<int> block_degrees;
} NewBlockDegrees;

typedef struct proposal_and_edge_counts_t {
    int proposal;
    int num_out_neighbor_edges;
    int num_in_neighbor_edges;
    int num_neighbor_edges;
} ProposalAndEdgeCounts;

/// TODO
int choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights);

/// TODO
int choose_neighbor(SparseVector<double> &multinomial_distribution);

/// Computes the block degrees under a proposed move
NewBlockDegrees compute_new_block_degrees(int current_block, Partition &partition, common::ProposalAndEdgeCounts &proposal);
/// TODO
double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree);

double delta_entropy_temp(MapVector<int> &row_or_col, std::vector<int> &block_degrees, int degree);

/// Removes entries from in whose index is index1 or index
std::vector<int> exclude_indices(std::vector<int> &in, int index1, int index2);

/// Removes entries from in whose index is index1 or index
MapVector<int>& exclude_indices(MapVector<int> &in, int index1, int index2);

/// Returns a subset of <values> corresponding to the indices where the value of <indices> != 0
std::vector<int> index_nonzero(std::vector<int> &values, std::vector<int> &indices);

/// Returns a subset of <values> corresponding to the indices where the value of <indices> != 0
std::vector<int> index_nonzero(std::vector<int> &values, MapVector<int> &indices);

/// Returns the non-zero values in <in>
std::vector<int> nonzeros(std::vector<int> &in);

/// Returns the non-zero values in <in>
std::vector<int> nonzeros(MapVector<int> &in);

/// TODO
ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                        std::vector<int> &block_partition, Partition &partition, bool block_merge = false);
/// TODO
int propose_random_block(int current_block, int num_blocks);

} // namespace common

#endif // SBP_COMMON_HPP
