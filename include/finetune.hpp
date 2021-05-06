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
EdgeWeights block_edge_weights(const std::vector<int> &block_assignment, EdgeWeights &neighbor_weights);
double compute_delta_entropy(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                             EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);
double compute_delta_entropy(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                             SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);
bool early_stop(int iteration, BlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies);
bool early_stop(int iteration, double initial_entropy, std::vector<double> &delta_entropies);
EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight);
void edge_count_updates_sparse(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight,
                               SparseEdgeCountUpdates &updates);
EdgeWeights edge_weights(const NeighborList &neighbors, int vertex);
double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           common::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);
double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           common::ProposalAndEdgeCounts &proposal, SparseEdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);
/// Computes the overall entropy of the given blockmodel.
double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges);
/// Proposes a new Metropolis-Hastings vertex move.
ProposalEvaluation propose_move(Blockmodel &blockmodel, int vertex, const Graph &graph);
//NeighborList &out_neighbors,
//                                const NeighborList &in_neighbors);
/// Proposes a new Asynchronous Gibbs vertex move.
VertexMove propose_gibbs_move(const Blockmodel &blockmodel, int vertex, const Graph &graph);
//NeighborList &out_neighbors,
//                              const NeighborList &in_neighbors);
Blockmodel &metropolis_hastings(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels);
Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels);
Blockmodel &finetune_assignment(Blockmodel &blockmodel, Graph &graph);

namespace directed {

/// Computes the overall entropy of the given blockmodel for a directed graph.
double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges);

}  // namespace directed

namespace undirected {

/// Computes the overall entropy of the given blockmodel for an undirected graph.
double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges);

}  // namespace undirected

namespace dist {

/// Runs the Asynchronous Gibbs algorithm in a distributed fashion using MPI.
Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels);

}  // namespace dist

} // namespace finetune

#endif // CPPSBP_FINETUNE_HPP
