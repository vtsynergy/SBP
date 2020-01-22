/**
 * The finetuning phase of the stochastic block partitioning algorithm.
 */
#ifndef CPPSBP_FINETUNE_HPP
#define CPPSBP_FINETUNE_HPP

#include <math.h>
#include <vector>

#include "common.hpp"
#include "partition/partition.hpp"
#include "partition/partition_triplet.hpp"

/*******************
 * FINE-TUNE
 ******************/
namespace finetune {

typedef struct proposal_evaluation_t {
    double delta_entropy;
    bool did_move;
} ProposalEvaluation;

static const int MOVING_AVG_WINDOW = 3;      // Window for calculating change in entropy
static const double SEARCH_THRESHOLD = 5e-4; // Threshold before golden ratio is established
static const double GOLDEN_THRESHOLD = 1e-4; // Threshold after golden ratio is established
static const int MAX_NUM_ITERATIONS = 100;   // Maximum number of finetuning iterations

bool accept(double delta_entropy, double hastings_correction);
EdgeWeights block_edge_weights(Vector &block_assignment, EdgeWeights &neighbor_weights);
double compute_delta_entropy(int current_block, int proposal, Partition &partition, EdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees);
bool early_stop(int iteration, PartitionTriplet &partitions, Partition &partition,
                std::vector<double> &delta_entropies);
EdgeCountUpdates edge_count_updates(BoostMappedMatrix &blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight);
EdgeWeights edge_weights(std::vector<Matrix2Column> &neighbors, int vertex);
double hastings_correction(Partition &partition, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           common::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);
double overall_entropy(Partition &partition, int num_vertices, int num_edges);
ProposalEvaluation propose_move(Partition &partition, int vertex, std::vector<Matrix2Column> &out_neighbors,
                                std::vector<Matrix2Column> &in_neighbors);
Partition &reassign_vertices(Partition &partition, int num_vertices, int num_edges,
                             std::vector<Matrix2Column> &out_neighbors, std::vector<Matrix2Column> &in_neighbors,
                             PartitionTriplet &partitions);

} // namespace finetune

#endif // CPPSBP_FINETUNE_HPP
