#include "finetune.hpp"

#include "../args.hpp"
#include "mpi_data.hpp"

#include "assert.h"

namespace finetune {

bool accept(double delta_entropy, double hastings_correction) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double random_probability = distribution(common::generator);
    double accept_probability = exp(-3.0 * delta_entropy) * hastings_correction;
    accept_probability = (accept_probability >= 1.0) ? 1.0 : accept_probability;
    return random_probability <= accept_probability;
}

Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels) {
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges()));
    double initial_entropy = blockmodel.getOverall_entropy();

    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        int num_batches = args.batches;
        int batch_size = int(ceil(graph.num_vertices() / num_batches));
        for (int batch = 0; batch < graph.num_vertices() / batch_size; ++batch) {
            int start = batch * batch_size;
            int end = std::min(graph.num_vertices(), (batch + 1) * batch_size);
            // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
            // asynchronous Gibbs sampling
            std::vector<int> block_assignment(blockmodel.block_assignment());
            #pragma omp parallel for schedule(dynamic)
            for (int vertex = start; vertex < end; ++vertex) {
                VertexMove proposal = propose_gibbs_move(blockmodel, vertex, graph);
                // VertexMove proposal = propose_gibbs_move(blockmodel, vertex, graph.out_neighbors(),
                //                                          graph.in_neighbors());
                if (proposal.did_move) {
                    vertex_moves++;
                    delta_entropy += proposal.delta_entropy;
                    block_assignment[vertex] = proposal.proposed_block;
                }
            }
            blockmodel = Blockmodel(blockmodel.getNum_blocks(), graph.out_neighbors(),
                                    blockmodel.getBlock_reduction_rate(), block_assignment);
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, blockmodels, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges()));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

EdgeWeights block_edge_weights(const std::vector<int> &block_assignment, EdgeWeights &neighbor_weights) {
    std::map<int, int> block_counts;
    for (uint i = 0; i < neighbor_weights.indices.size(); ++i) {
        int neighbor = neighbor_weights.indices[i];
        int block = block_assignment[neighbor];
        int weight = neighbor_weights.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    std::vector<int> blocks;
    std::vector<int> weights;
    for (auto const &entry : block_counts) {
        blocks.push_back(entry.first);
        weights.push_back(entry.second);
    }
    return EdgeWeights{blocks, weights};
}

double compute_delta_entropy(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                             EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<int> old_block_row = blockmodel.blockmatrix()->getrow(current_block); // M_r_t1
    std::vector<int> old_proposal_row = blockmodel.blockmatrix()->getrow(proposal);   // M_s_t1
    std::vector<int> old_block_col = blockmodel.blockmatrix()->getcol(current_block); // M_t2_r
    std::vector<int> old_proposal_col = blockmodel.blockmatrix()->getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<int> new_block_col = common::exclude_indices(updates.block_col, current_block, proposal); // added
    std::vector<int> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<int> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block, proposal);
    std::vector<int> old_block_degrees_out = common::exclude_indices(blockmodel.getBlock_degrees_out(), current_block, proposal);

    // Remove 0 indices
    std::vector<int> new_block_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in, updates.block_row); // added
    std::vector<int> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in, updates.proposal_row);
    std::vector<int> new_block_row = common::nonzeros(updates.block_row); // added
    std::vector<int> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<int> new_block_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_block_col); // added
    std::vector<int> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_block_col = common::nonzeros(new_block_col); // added
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<int> old_block_row_degrees_in = common::index_nonzero(blockmodel.getBlock_degrees_in(), old_block_row);
    std::vector<int> old_proposal_row_degrees_in = common::index_nonzero(blockmodel.getBlock_degrees_in(), old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<int> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<int> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_block_row, new_block_row_degrees_in,
                                                block_degrees.block_degrees_out[current_block], num_edges); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(new_block_col, new_block_col_degrees_out,
                                                block_degrees.block_degrees_in[current_block], num_edges); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                blockmodel.getBlock_degrees_out()[current_block], num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                blockmodel.getBlock_degrees_out()[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                blockmodel.getBlock_degrees_in()[current_block], num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                blockmodel.getBlock_degrees_in()[proposal], num_edges);
    if (std::isnan(delta_entropy)) {
        std::cout << "===================ARGAGDJAKLJDAJFKLDJA" << std::endl;
        exit(-142321);
    }
    return delta_entropy;
}

double compute_delta_entropy(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                             SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const ISparseMatrix *matrix = blockmodel.blockmatrix();
    const MapVector<int> &old_block_row = matrix->getrow_sparse(current_block); // M_r_t1
    const MapVector<int> &old_proposal_row = matrix->getrow_sparse(proposal);   // M_s_t1
    const MapVector<int> &old_block_col = matrix->getcol_sparse(current_block); // M_t2_r
    const MapVector<int> &old_proposal_col = matrix->getcol_sparse(proposal);   // M_t2_s

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(updates.block_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[current_block], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.proposal_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.block_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[current_block], current_block, proposal,
                                                num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.proposal_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[proposal], current_block, proposal,
                                                num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_block_row, blockmodel.getBlock_degrees_in(),
                                                blockmodel.getBlock_degrees_out()[current_block], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_proposal_row, blockmodel.getBlock_degrees_in(),
                                                blockmodel.getBlock_degrees_out()[proposal], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_block_col, blockmodel.getBlock_degrees_out(),
                                                blockmodel.getBlock_degrees_in()[current_block], current_block,
                                                proposal, num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_proposal_col, blockmodel.getBlock_degrees_out(),
                                                blockmodel.getBlock_degrees_in()[proposal], current_block, proposal,
                                                num_edges);
    assert(!std::isnan(delta_entropy));
    if (std::isnan(delta_entropy)) {
        std::cout << "ARGAGDJAKLJDAJFKLDJA" << std::endl;
        exit(-142321);
    }
    return delta_entropy;
}

bool early_stop(int iteration, BlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies) {
    int last_index = delta_entropies.size() - 1;
    if (delta_entropies[last_index] == 0.0) {
        return true;
    }
    if (iteration < 3) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (blockmodels.get(2).empty) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return (average < threshold) ? true : false;
}

bool early_stop(int iteration, double initial_entropy, std::vector<double> &delta_entropies) {
    if (iteration < 3) {
        return false;
    }
    int last_index = delta_entropies.size() - 1;
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold = 1e-4 * initial_entropy;
    return (average < threshold) ? true : false;
}

EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight) {
    std::vector<int> block_row = blockmodel->getrow(current_block);
    std::vector<int> block_col = blockmodel->getcol(current_block);
    std::vector<int> proposal_row = blockmodel->getrow(proposed_block);
    std::vector<int> proposal_col = blockmodel->getcol(proposed_block);

    int count_in_block = 0, count_out_block = 0;
    int count_in_proposal = self_edge_weight, count_out_proposal = self_edge_weight;

    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int index = in_blocks.indices[i];
        int value = in_blocks.values[i];
        if (index == current_block) {
            count_in_block += value;
        }
        if (index == proposed_block) {
            count_in_proposal += value;
        }
        block_col[index] -= value;
        proposal_col[index] += value;
    }
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int index = out_blocks.indices[i];
        int value = out_blocks.values[i];
        if (index == current_block) {
            count_out_block += value;
        }
        if (index == proposed_block) {
            count_out_proposal += value;
        }
        block_row[index] -= value;
        proposal_row[index] += value;
    }

    proposal_row[current_block] -= count_in_proposal;
    proposal_row[proposed_block] += count_in_proposal;
    proposal_col[current_block] -= count_out_proposal;
    proposal_col[proposed_block] += count_out_proposal;

    block_row[current_block] -= count_in_block;
    block_row[proposed_block] += count_in_block;
    block_col[current_block] -= count_out_block;
    block_col[proposed_block] += count_out_block;

    return EdgeCountUpdates{block_row, proposal_row, block_col, proposal_col};
}

void edge_count_updates_sparse(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight,
                               SparseEdgeCountUpdates &updates) {
    updates.block_row = blockmodel->getrow_sparse(current_block);
    updates.block_col = blockmodel->getcol_sparse(current_block);
    updates.proposal_row = blockmodel->getrow_sparse(proposed_block);
    updates.proposal_col = blockmodel->getcol_sparse(proposed_block);

    int count_in_block = 0, count_out_block = 0;
    int count_in_proposal = self_edge_weight, count_out_proposal = self_edge_weight;

    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int index = in_blocks.indices[i];
        int value = in_blocks.values[i];
        if (index == current_block) {
            count_in_block += value;
        }
        if (index == proposed_block) {
            count_in_proposal += value;
        }
        updates.block_col[index] -= value;
        updates.proposal_col[index] += value;
    }
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int index = out_blocks.indices[i];
        int value = out_blocks.values[i];
        if (index == current_block) {
            count_out_block += value;
        }
        if (index == proposed_block) {
            count_out_proposal += value;
        }
        updates.block_row[index] -= value;
        updates.proposal_row[index] += value;
    }

    updates.proposal_row[current_block] -= count_in_proposal;
    updates.proposal_row[proposed_block] += count_in_proposal;
    updates.proposal_col[current_block] -= count_out_proposal;
    updates.proposal_col[proposed_block] += count_out_proposal;

    updates.block_row[current_block] -= count_in_block;
    updates.block_row[proposed_block] += count_in_block;
    updates.block_col[current_block] -= count_out_block;
    updates.block_col[proposed_block] += count_out_block;
}

EdgeWeights edge_weights(const NeighborList &neighbors, int vertex) {
    std::vector<int> indices;
    std::vector<int> values;
    // Assumes graph is unweighted
    const std::vector<int> &neighbor_vector = neighbors[vertex];
    for (int row = 0; row < neighbor_vector.size(); ++row) {
        indices.push_back(neighbor_vector[row]);
        values.push_back(1);
    }
    return EdgeWeights{indices, values};
}

double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           common::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees) {
    if (proposal.num_neighbor_edges == 0) {
        return 1.0;
    }
    // Compute block weights
    std::map<int, int> block_counts;
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int block = out_blocks.indices[i];
        int weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int block = in_blocks.indices[i];
        int weight = in_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    // Create Arrays using unique blocks
    int num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<int> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<int> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    int index = 0;
    int num_blocks = blockmodel.getNum_blocks();
    const std::vector<int> &current_block_degrees = blockmodel.getBlock_degrees();
    for (auto const &entry : block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = proposal_row[entry.first] + proposal_col[entry.first] + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = updates.block_row[entry.first] + updates.block_col[entry.first] + 1.0;
        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    double p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    double p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           common::ProposalAndEdgeCounts &proposal, SparseEdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees) {
    if (proposal.num_neighbor_edges == 0) {
        return 1.0;
    }
    // Compute block weights
    std::map<int, int> block_counts;
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int block = out_blocks.indices[i];
        int weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int block = in_blocks.indices[i];
        int weight = in_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    // Create Arrays using unique blocks
    int num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<int> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<int> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    int index = 0;
    int num_blocks = blockmodel.getNum_blocks();
    const std::vector<int> &current_block_degrees = blockmodel.getBlock_degrees();
    for (auto const &entry : block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = proposal_row[entry.first] + proposal_col[entry.first] + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = updates.block_row[entry.first] + updates.block_col[entry.first] + 1.0;
        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    double p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    double p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges) {
    if (args.undirected)
        return undirected::overall_entropy(blockmodel, num_vertices, num_edges);
    return directed::overall_entropy(blockmodel, num_vertices, num_edges);
}

ProposalEvaluation propose_move(Blockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);  // getBlock_assignment()[vertex];
    EdgeWeights vertex_out_neighbors = edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights vertex_in_neighbors = edge_weights(graph.in_neighbors(), vertex);

    common::ProposalAndEdgeCounts proposal = common::propose_new_block(
        current_block, vertex_out_neighbors, vertex_in_neighbors, blockmodel.block_assignment(), blockmodel, false);
    if (proposal.proposal == current_block) {
        return ProposalEvaluation{0.0, did_move};
    }

    EdgeWeights blocks_out_neighbors = block_edge_weights(blockmodel.block_assignment(), vertex_out_neighbors);
    EdgeWeights blocks_in_neighbors = block_edge_weights(blockmodel.block_assignment(), vertex_in_neighbors);
    int self_edge_weight = 0;
    for (uint i = 0; i < vertex_out_neighbors.indices.size(); ++i) {
        if (vertex_out_neighbors.indices[i] == vertex) {
            self_edge_weight = vertex_out_neighbors.values[i];
            break;
        }
    }

    // TODO: change this to sparse_edge_count_updates
    EdgeCountUpdates updates = edge_count_updates(blockmodel.blockmatrix(), current_block, proposal.proposal,
                                                  blocks_out_neighbors, blocks_in_neighbors, self_edge_weight);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, blockmodel, proposal);
    double hastings =
        hastings_correction(blockmodel, blocks_out_neighbors, blocks_in_neighbors, proposal, updates, new_block_degrees);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, blockmodel, graph.num_edges(), updates,
                              new_block_degrees);
    if (accept(delta_entropy, hastings)) {
        blockmodel.move_vertex(vertex, current_block, proposal.proposal, updates, new_block_degrees.block_degrees_out,
                              new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
        did_move = true;
    }
    return ProposalEvaluation{delta_entropy, did_move};
}

VertexMove propose_gibbs_move(const Blockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);
    EdgeWeights vertex_out_neighbors = edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights vertex_in_neighbors = edge_weights(graph.in_neighbors(), vertex);

    common::ProposalAndEdgeCounts proposal = common::propose_new_block(
        current_block, vertex_out_neighbors, vertex_in_neighbors, blockmodel.block_assignment(), blockmodel, false);
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1};
    }

    EdgeWeights blocks_out_neighbors = block_edge_weights(blockmodel.block_assignment(), vertex_out_neighbors);
    EdgeWeights blocks_in_neighbors = block_edge_weights(blockmodel.block_assignment(), vertex_in_neighbors);
    int self_edge_weight = 0;
    for (uint i = 0; i < vertex_out_neighbors.indices.size(); ++i) {
        if (vertex_out_neighbors.indices[i] == vertex) {
            self_edge_weight = vertex_out_neighbors.values[i];
            break;
        }
    }

    SparseEdgeCountUpdates updates;
    edge_count_updates_sparse(blockmodel.blockmatrix(), current_block, proposal.proposal, blocks_out_neighbors,
                              blocks_in_neighbors, self_edge_weight, updates);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, blockmodel, proposal);
    double hastings =
        hastings_correction(blockmodel, blocks_out_neighbors, blocks_in_neighbors, proposal, updates, new_block_degrees);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, blockmodel, graph.num_edges(), updates,
                              new_block_degrees);
    if (accept(delta_entropy, hastings)) {
        did_move = true;
        return VertexMove{delta_entropy, did_move, vertex, proposal.proposal};
    }
    return VertexMove{delta_entropy, did_move, -1, -1};
}

Blockmodel &metropolis_hastings(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels) {
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges()));
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            ProposalEvaluation proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, blockmodels, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges()));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

Blockmodel &finetune_assignment(Blockmodel &blockmodel, Graph &graph) {
    std::vector<double> delta_entropies;
    // TODO: Add number of finetuning iterations to evaluation
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges()));
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            ProposalEvaluation proposal = propose_move(blockmodel, vertex, graph);
            // ProposalEvaluation proposal = propose_move(blockmodel, vertex, graph.out_neighbors(),
            //                                            graph.in_neighbors());
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of finetuning moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / blockmodel.getOverall_entropy() << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges()));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

namespace directed {

double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

}  // namespace directed

namespace undirected {

double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges) {
    std::cout << "undirected!" << std::endl;
    double log_posterior_p = blockmodel.log_posterior_probability(num_edges);
    if (std::isnan(log_posterior_p)) {
        std::cout << "nan in log posterior" << std::endl;
        exit(-5000);
    }
    double x = blockmodel.getNum_blocks() * (blockmodel.getNum_blocks() + 1.0) / (2.0 * num_edges);
    if (std::isnan(x)) {
        std::cout << "nan in X" << std::endl;
        exit(-5000);
    }
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    if (std::isnan(h)) {
        std::cout << "nan in h()" << std::endl;
    }
        // std::cout << "X: " << x << std::endl;
        // std::cout << "log(X): " << log(x) << std::endl;
    if (std::isnan(h)) {
        exit(-5000);
    }
    double first = (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks()));
    std::cout << "first: " << first << " log_posterior: " << log_posterior_p << std::endl;
    double result = (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
    if (std::isnan(result)) {
        std::cout << "nan in result" << std::endl;
        exit(-5000);
    }
    return result;
}

}  // namespace undirected

namespace dist {

TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    // MPI Datatype init
    MPI_Datatype Membership_t;
    int membership_blocklengths[2] = { 1, 1 };
    MPI_Aint membership_displacements[2] = { 0, sizeof(int) };
    MPI_Datatype membership_types[2] = { MPI_INT, MPI_INT };
    MPI_Type_create_struct(2, membership_blocklengths, membership_displacements, membership_types, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    double old_entropy = dist::overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
    double initial_entropy = blockmodel.getOverall_entropy();
    double new_entropy = 0;

    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        int num_batches = args.batches;
        int batch_size = int(ceil(graph.num_vertices() / num_batches));
        // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
        // asynchronous Gibbs sampling
        std::vector<int> block_assignment(blockmodel.block_assignment());
        for (int batch = 0; batch < graph.num_vertices() / batch_size; ++batch) {
            int start = batch * batch_size;
            int end = std::min(graph.num_vertices(), (batch + 1) * batch_size);
            std::vector<Membership> membership_updates;
            #pragma omp parallel for schedule(dynamic)
            // for (int vertex = start + mpi.rank; vertex < end; vertex += mpi.num_processes) {
            for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
                int block = blockmodel.block_assignment(vertex);
                if (!(block % mpi.num_processes == mpi.rank))
                    continue;
                VertexMove proposal = dist::propose_gibbs_move(blockmodel, vertex, graph);
                if (proposal.did_move) {
                // std::cout << "proposal.proposed_block: " << proposal.proposed_block << " size: " << blockmodel.in_two_hop_radius().size() << std::endl;
                    assert(blockmodel.owns(proposal.proposed_block));
                    membership_updates.push_back(Membership { vertex, proposal.proposed_block });
                }
            }
            int num_moves = membership_updates.size();
            // MPI COMMUNICATION
            int rank_moves[mpi.num_processes];
            MPI_Allgather(&num_moves, 1, MPI_INT, &rank_moves, 1, MPI_INT, MPI_COMM_WORLD);
            int offsets[mpi.num_processes];
            offsets[0] = 0;
            for (int i = 1; i < mpi.num_processes; ++i) {
                offsets[i] = offsets[i-1] + rank_moves[i-1];
            }
            int batch_vertex_moves = offsets[mpi.num_processes-1] + rank_moves[mpi.num_processes-1];
            std::vector<Membership> collected_membership_updates(batch_vertex_moves);
            MPI_Allgatherv(membership_updates.data(), num_moves, Membership_t, collected_membership_updates.data(),
                           rank_moves, offsets, Membership_t, MPI_COMM_WORLD);
            // END MPI COMMUNICATION
            for (const Membership &membership : collected_membership_updates) {
                block_assignment[membership.vertex] = membership.block;
            }
            blockmodel = TwoHopBlockmodel(blockmodel.getNum_blocks(), graph.out_neighbors(),
                                          blockmodel.getBlock_reduction_rate(), block_assignment);
            vertex_moves += batch_vertex_moves;
        }
        new_entropy = dist::overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / initial_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, blockmodels, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    return blockmodel;
}

bool early_stop(int iteration, DistBlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies) {
    int last_index = delta_entropies.size() - 1;
    if (delta_entropies[last_index] == 0.0) {
        return true;
    }
    if (iteration < 3) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (blockmodels.get(2).empty) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return (average < threshold) ? true : false;
}

double overall_entropy(const TwoHopBlockmodel &blockmodel, int num_vertices, int num_edges) {
    return dist::directed::overall_entropy(blockmodel, num_vertices, num_edges);
    // if (args.undirected)
    //     return dist::undirected::overall_entropy(blockmodel, num_vertices, num_edges);
    // return dist::directed::overall_entropy(blockmodel, num_vertices, num_edges);
}

VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);
    EdgeWeights vertex_out_neighbors = edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights vertex_in_neighbors = edge_weights(graph.in_neighbors(), vertex);

    common::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
        current_block, vertex_out_neighbors, vertex_in_neighbors, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.owns(proposal.proposal)) {
        std::cerr << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1};
    }

    EdgeWeights blocks_out_neighbors = block_edge_weights(blockmodel.block_assignment(), vertex_out_neighbors);
    EdgeWeights blocks_in_neighbors = block_edge_weights(blockmodel.block_assignment(), vertex_in_neighbors);
    int self_edge_weight = 0;
    for (uint i = 0; i < vertex_out_neighbors.indices.size(); ++i) {
        if (vertex_out_neighbors.indices[i] == vertex) {
            self_edge_weight = vertex_out_neighbors.values[i];
            break;
        }
    }

    SparseEdgeCountUpdates updates;
    edge_count_updates_sparse(blockmodel.blockmatrix(), current_block, proposal.proposal, blocks_out_neighbors,
                              blocks_in_neighbors, self_edge_weight, updates);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, blockmodel, proposal);
    double hastings =
        hastings_correction(blockmodel, blocks_out_neighbors, blocks_in_neighbors, proposal, updates, new_block_degrees);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, blockmodel, graph.num_edges(), updates,
                              new_block_degrees);
    if (accept(delta_entropy, hastings)) {
        did_move = true;
        return VertexMove{delta_entropy, did_move, vertex, proposal.proposal};
    }
    return VertexMove{delta_entropy, did_move, -1, -1};
}

namespace directed {

double overall_entropy(const TwoHopBlockmodel &blockmodel, int num_vertices, int num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

}  // namespace directed

}  // namespace dist

}  // namespace finetune