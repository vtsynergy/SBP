/**
 * Structs and functions common to both the block merge and finetune phases.
 */

#ifndef SBP_COMMON_HPP
#define SBP_COMMON_HPP

#include <random>
#include <vector>

// #include <Eigen/Core>

#include "blockmodel/blockmodel.hpp"
#include "blockmodel/sparse/csparse_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "utils.hpp"

namespace common {

// static std::random_device seeder;
// static std::mt19937_64 generator(seeder());
const int seed = 1;
static std::mt19937_64 generator(seed);

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

int choose_neighbor_uniform(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights);

/// Chooses a neighboring block using a multinomial distribution based on the number of edges connecting the current
/// block to the neighboring blocks.
int choose_neighbor(const SparseVector<double> &multinomial_distribution);

int choose_neighbor_uniform(const SparseVector<double> &multinomial_distribution);

/// Computes the block degrees under a proposed move
NewBlockDegrees compute_new_block_degrees(int current_block, Blockmodel &blockmodel,
                                          common::ProposalAndEdgeCounts &proposal);

/// Computes the entropy of one row or column of data
double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree);

/// Computes the entropy of one row or column of sparse data
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree);

/// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal);

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

/// Proposes a new block for either the block merge or finetune step based on `bool block_merge`.
ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                        std::vector<int> &block_blockmodel, Blockmodel &blockmodel, bool block_merge = false);

/// Proposes a new block for the block merge phase based on `bool block_merge`. Selects a neighboring block of a
/// neighboring block. If random == true, a random block is proposed.
ProposalAndEdgeCounts propose_new_block(int current_block, Blockmodel &blockmodel, bool random);

/// Proposes a new block for the mcmc phase. Selects a neighboring block of a neighboring vertex.
ProposalAndEdgeCounts propose_new_block(int vertex, EdgeWeights &out_vertices, EdgeWeights &in_vertices,
                                        Blockmodel &blockmodel);

/// Proposes a new block for either the block merge or finetune step based on `bool block_merge`.
ProposalAndEdgeCounts propose_new_block_mcmc(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                             std::vector<int> &block_blockmodel, Blockmodel &blockmodel,
                                             bool block_merge = false);
/// TODO
int propose_random_block(int current_block, int num_blocks);

/// Samples a random neighbor of a vertex, using its out_edges and in_edges.
int random_neighbor(EdgeWeights &out_edges, EdgeWeights &in_edges);

} // namespace common

#endif // SBP_COMMON_HPP
