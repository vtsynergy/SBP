/**
 * The finetuning phase of the stochastic block blockmodeling algorithm.
 */
#ifndef SBP_FINETUNE_HPP
#define SBP_FINETUNE_HPP

#include <math.h>
#include <vector>

#include <omp.h>

// #include <boost/variant.hpp>

#include "common.hpp"
#include "graph.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "utils.hpp"

/*******************
 * FINE-TUNE
 ******************/
namespace finetune {

typedef struct proposal_evaluation_t {
    double delta_entropy;
    bool did_move;
} ProposalEvaluation;

typedef struct vertex_move_t {
    double delta_entropy;
    bool did_move;
    int vertex;
    int proposed_block;
} VertexMove;

struct Proposal {
    ProposalEvaluation eval;
    VertexMove move;
};

static const int MOVING_AVG_WINDOW = 3;      // Window for calculating change in entropy
static const double SEARCH_THRESHOLD = 5e-4; // Threshold before golden ratio is established
static const double GOLDEN_THRESHOLD = 1e-4; // Threshold after golden ratio is established
static const int MAX_NUM_ITERATIONS = 100;   // Maximum number of finetuning iterations

bool accept(double delta_entropy, double hastings_correction);
EdgeWeights block_edge_weights(std::vector<int> &block_assignment, EdgeWeights &neighbor_weights);
double compute_delta_entropy(int current_block, int proposal, Blockmodel &blockmodel, EdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees);
double compute_delta_entropy(int current_block, int proposal, Blockmodel &blockmodel, SparseEdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees);
bool early_stop(int iteration, BlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies);
bool early_stop(int iteration, double initial_entropy, std::vector<double> &delta_entropies);
EdgeCountUpdates edge_count_updates(DictTransposeMatrix &blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight);
void edge_count_updates_sparse(DictTransposeMatrix &blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight,
                               SparseEdgeCountUpdates &updates);
EdgeWeights edge_weights(NeighborList &neighbors, int vertex);
double hastings_correction(Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           common::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);
double hastings_correction(Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           common::ProposalAndEdgeCounts &proposal, SparseEdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);
double overall_entropy(Blockmodel &blockmodel, int num_vertices, int num_edges);
ProposalEvaluation propose_move(Blockmodel &blockmodel, int vertex, NeighborList &out_neighbors,
                                NeighborList &in_neighbors);
VertexMove propose_gibbs_move(Blockmodel &blockmodel, int vertex, NeighborList &out_neighbors,
                            NeighborList &in_neighbors);
Blockmodel &metropolis_hastings(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels, Args &args);
Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels, Args &args);
Blockmodel &finetune_assignment(Blockmodel &blockmodel, Graph &graph);

} // namespace finetune

#endif // CPPSBP_FINETUNE_HPP
