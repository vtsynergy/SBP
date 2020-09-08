#include "finetune.hpp"

// Reads: NA
// Writes: NA
bool finetune::accept(double delta_entropy, double hastings_correction) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double random_probability = distribution(common::generator);
    double accept_probability = exp(-3.0 * delta_entropy) * hastings_correction;
    accept_probability = (accept_probability >= 1.0) ? 1.0 : accept_probability;
    return random_probability <= accept_probability;
}

// Reads: NA
// Writes: NA
EdgeWeights finetune::block_edge_weights(std::vector<int> &block_assignment, EdgeWeights &neighbor_weights) {
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

// Reads: 
//   - partition.blockmodel current_block row
//   - partition.blockmodel current_block col
//   - partition.blockmodel proposal row
//   - partition.blockmodel proposal col
//   - partition.block_degrees_out
//   - partition.block_degrees_in
// Writes: NA
double finetune::compute_delta_entropy(int current_block, int proposal, Partition &partition, EdgeCountUpdates &updates,
                                       common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<int> old_block_row = partition.getBlockmodel().getrow(current_block); // M_r_t1
    std::vector<int> old_proposal_row = partition.getBlockmodel().getrow(proposal);   // M_s_t1
    std::vector<int> old_block_col = partition.getBlockmodel().getcol(current_block); // M_t2_r
    std::vector<int> old_proposal_col = partition.getBlockmodel().getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<int> new_block_col = common::exclude_indices(updates.block_col, current_block, proposal); // added
    std::vector<int> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<int> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block, proposal);
    std::vector<int> old_block_degrees_out = common::exclude_indices(partition.getBlock_degrees_out(), current_block, proposal);

    // Remove 0 indices
    std::vector<int> new_block_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in, updates.block_row); // added
    std::vector<int> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in, updates.proposal_row);
    std::vector<int> new_block_row = common::nonzeros(updates.block_row); // added
    std::vector<int> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<int> new_block_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_block_col); // added
    std::vector<int> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_block_col = common::nonzeros(new_block_col); // added
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<int> old_block_row_degrees_in = common::index_nonzero(partition.getBlock_degrees_in(), old_block_row);
    std::vector<int> old_proposal_row_degrees_in = common::index_nonzero(partition.getBlock_degrees_in(), old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<int> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<int> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_block_row, new_block_row_degrees_in,
                                                block_degrees.block_degrees_out[current_block]); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal]);
    delta_entropy -= common::delta_entropy_temp(new_block_col, new_block_col_degrees_out,
                                                block_degrees.block_degrees_in[current_block]); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal]);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                partition.getBlock_degrees_out()[current_block]);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                partition.getBlock_degrees_out()[proposal]);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                partition.getBlock_degrees_in()[current_block]);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                partition.getBlock_degrees_in()[proposal]);
    return delta_entropy;
}

// Reads:
//   - partition.overall_entropy
// Writes: NA
bool finetune::early_stop(int iteration, PartitionTriplet &partitions, double initial_entropy,
                          std::vector<double> &delta_entropies) {
    if (iteration < 3) {
        return false;
    }
    int last_index = delta_entropies.size() - 1;
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (partitions.get(2).empty) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return (average < threshold) ? true : false;
}

// Reads:
//   - partition.overall_entropy
// Writes: NA
bool finetune::early_stop(int iteration, double initial_entropy, std::vector<double> &delta_entropies) {
    if (iteration < 3) {
        return false;
    }
    int last_index = delta_entropies.size() - 1;
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold = 1e-4 * initial_entropy;
    return (average < threshold) ? true : false;
}

// Reads:
//   - blockmodel current_block row
//   - blockmodel current_block col
//   - blockmodel proposed_block row
//   - blockmodel proposed_block col
// Writes: NA
EdgeCountUpdates finetune::edge_count_updates(DictMatrix &blockmodel, int current_block, int proposed_block,
                                              EdgeWeights &out_blocks, EdgeWeights &in_blocks, int self_edge_weight) {
    std::vector<int> block_row = blockmodel.getrow(current_block);
    std::vector<int> block_col = blockmodel.getcol(current_block);
    std::vector<int> proposal_row = blockmodel.getrow(proposed_block);
    std::vector<int> proposal_col = blockmodel.getcol(proposed_block);

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

// Reads: NA
// Writes: NA
/// Returns the edge weights for the neighbors of a given vertex.
/// Assumes the graph is unweighted, so all edge weights are 1.
EdgeWeights finetune::edge_weights(NeighborList &neighbors, int vertex) {
    std::vector<int> indices;
    std::vector<int> values;
    // Assumes graph is undirected
    std::vector<int> neighbor_vector = neighbors[vertex];
    for (int row = 0; row < neighbor_vector.size(); ++row) {
        indices.push_back(neighbor_vector[row]);
        values.push_back(1);
    }
    return EdgeWeights{indices, values};
}

// Reads:
//   - partition.blockmodel proposal row
//   - partition.blockmodel proposal col
//   - partition.num_blocks
//   - partition.block_degrees
// Writes: NA
/// TODO
double finetune::hastings_correction(Partition &partition, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
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
    // std::vector<double> counts = utils::constant<double>(num_unique_blocks, 0);
    // std::vector<double> proposal_weights = utils::constant<double>(num_unique_blocks, 0);
    // std::vector<double> block_weights = utils::constant<double>(num_unique_blocks, 0);
    // std::vector<double> block_degrees = utils::constant<double>(num_unique_blocks, 0);
    // std::vector<double> proposal_degrees = utils::constant<double>(num_unique_blocks, 0);    
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<int> proposal_row = partition.getBlockmodel().getrow(proposal.proposal);
    std::vector<int> proposal_col = partition.getBlockmodel().getcol(proposal.proposal);
    // Fill Arrays
    int index = 0;
    int num_blocks = partition.getNum_blocks();
    std::vector<int> &current_block_degrees = partition.getBlock_degrees();
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



// Reads:
//   - partition.blockmodel nonzero indices
//   - partition.blockmodel nonzero values
//   - partition degrees in
//   - partition degrees out
//   - partition.num_blocks
// Writes: NA
/// TODO
double finetune::overall_entropy(Partition &partition, int num_vertices, int num_edges) {
    double log_posterior_p = partition.log_posterior_probability();
    double x = pow(partition.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(partition.getNum_blocks())) - log_posterior_p;
}

// Reads:
//   - partition.block_assignment
//   - partition.num_blocks
//   - partition.blockmodel random_neighbor row
//   - partition.blockmodel random_neighbor col
//   - partition.blockmodel current_block row
//   - partition.blockmodel current_block col
//   - partition.blockmodel proposed_block row
//   - partition.blockmodel proposed_block col
//   - partition.block_degrees_out
//   - partition.block_degrees_in
//   - partition.block_degrees
// Writes:
//   - partition.block_assignment vertex
//   - partition.block_degrees_out
//   - partition.block_degrees_in
//   - partition.block_degrees
//   - partition.blockmodel current_block row
//   - partition.blockmodel current_block col
//   - partition.blockmodel proposed_block row
//   - partition.blockmodel proposed_block col
/// TODO
finetune::ProposalEvaluation finetune::propose_move(Partition &partition, int vertex,
                                                    NeighborList &out_neighbors, NeighborList &in_neighbors) {
    bool did_move = false;
    int current_block = partition.getBlock_assignment()[vertex];
    EdgeWeights vertex_out_neighbors = edge_weights(out_neighbors, vertex);
    EdgeWeights vertex_in_neighbors = edge_weights(in_neighbors, vertex);

    common::ProposalAndEdgeCounts proposal = common::propose_new_block(
        current_block, vertex_out_neighbors, vertex_in_neighbors, partition.getBlock_assignment(), partition, false);
    if (proposal.proposal == current_block) {
        return ProposalEvaluation{0.0, did_move};
    }

    EdgeWeights blocks_out_neighbors = block_edge_weights(partition.getBlock_assignment(), vertex_out_neighbors);
    EdgeWeights blocks_in_neighbors = block_edge_weights(partition.getBlock_assignment(), vertex_in_neighbors);
    int self_edge_weight = 0;
    for (uint i = 0; i < vertex_out_neighbors.indices.size(); ++i) {
        if (vertex_out_neighbors.indices[i] == vertex) {
            self_edge_weight = vertex_out_neighbors.values[i];
            break;
        }
    }

    EdgeCountUpdates updates = edge_count_updates(partition.getBlockmodel(), current_block, proposal.proposal,
                                                  blocks_out_neighbors, blocks_in_neighbors, self_edge_weight);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, partition, proposal);
    double hastings =
        hastings_correction(partition, blocks_out_neighbors, blocks_in_neighbors, proposal, updates, new_block_degrees);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, partition, updates, new_block_degrees);
    if (accept(delta_entropy, hastings)) {
        partition.move_vertex(vertex, current_block, proposal.proposal, updates, new_block_degrees.block_degrees_out,
                              new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
        did_move = true;
    }
    return ProposalEvaluation{delta_entropy, did_move};
}

/// TODO
finetune::VertexMove finetune::propose_gibbs_move(Partition &partition, int vertex,
                                                  NeighborList &out_neighbors, NeighborList &in_neighbors) {
    bool did_move = false;
    int current_block = partition.getBlock_assignment()[vertex];
    EdgeWeights vertex_out_neighbors = edge_weights(out_neighbors, vertex);
    EdgeWeights vertex_in_neighbors = edge_weights(in_neighbors, vertex);

    common::ProposalAndEdgeCounts proposal = common::propose_new_block(
        current_block, vertex_out_neighbors, vertex_in_neighbors, partition.getBlock_assignment(), partition, false);
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1};
    }

    EdgeWeights blocks_out_neighbors = block_edge_weights(partition.getBlock_assignment(), vertex_out_neighbors);
    EdgeWeights blocks_in_neighbors = block_edge_weights(partition.getBlock_assignment(), vertex_in_neighbors);
    int self_edge_weight = 0;
    for (uint i = 0; i < vertex_out_neighbors.indices.size(); ++i) {
        if (vertex_out_neighbors.indices[i] == vertex) {
            self_edge_weight = vertex_out_neighbors.values[i];
            break;
        }
    }

    EdgeCountUpdates updates = edge_count_updates(partition.getBlockmodel(), current_block, proposal.proposal,
                                                  blocks_out_neighbors, blocks_in_neighbors, self_edge_weight);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, partition, proposal);
    double hastings =
        hastings_correction(partition, blocks_out_neighbors, blocks_in_neighbors, proposal, updates, new_block_degrees);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, partition, updates, new_block_degrees);
    if (accept(delta_entropy, hastings)) {
        did_move = true;
        return VertexMove{delta_entropy, did_move, vertex, proposal.proposal};
    }
    return VertexMove{delta_entropy, did_move, -1, -1};
}

// Reads:
//   - partition.block_assignment
//   - partition.num_blocks
//   - partition.blockmodel random_neighbor row
//   - partition.blockmodel random_neighbor col
//   - partition.blockmodel current_block row
//   - partition.blockmodel current_block col
//   - partition.blockmodel proposed_block row
//   - partition.blockmodel proposed_block col
//   - partition.block_degrees_out
//   - partition.block_degrees_in
//   - partition.block_degrees
// Writes:
//   - partition.block_assignment vertex
//   - partition.block_degrees_out
//   - partition.block_degrees_in
//   - partition.block_degrees
//   - partition.blockmodel current_block row
//   - partition.blockmodel current_block col
//   - partition.blockmodel proposed_block row
//   - partition.blockmodel proposed_block col
//   - partition.overall_entropy
/// TODO
Partition &finetune::reassign_vertices(Partition &partition, Graph &graph, PartitionTriplet &partitions) {
    if (partition.getNum_blocks() == 1) {
        return partition;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    partition.setOverall_entropy(overall_entropy(partition, graph.num_vertices, graph.num_edges));
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
            ProposalEvaluation proposal = propose_move(partition, vertex, graph.out_neighbors,
                                                       graph.in_neighbors);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / partition.getOverall_entropy() << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, partitions, partition.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    partition.setOverall_entropy(overall_entropy(partition, graph.num_vertices, graph.num_edges));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << partition.getOverall_entropy() << std::endl;
    return partition;
}

/// TODO
Partition &finetune::finetune_assignment(Partition &partition, Graph &graph) {
    std::vector<double> delta_entropies;
    // TODO: Add number of finetuning iterations to evaluation
    int total_vertex_moves = 0;
    partition.setOverall_entropy(overall_entropy(partition, graph.num_vertices, graph.num_edges));
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
            ProposalEvaluation proposal = propose_move(partition, vertex, graph.out_neighbors,
                                                       graph.in_neighbors);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of finetuning moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / partition.getOverall_entropy() << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, partition.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    partition.setOverall_entropy(overall_entropy(partition, graph.num_vertices, graph.num_edges));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << partition.getOverall_entropy() << std::endl;
    return partition;
}

/// TODO
Partition &finetune::asynchronous_gibbs(Partition &partition, Graph &graph, PartitionTriplet &partitions) {
    if (partition.getNum_blocks() == 1) {
        return partition;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    partition.setOverall_entropy(overall_entropy(partition, graph.num_vertices, graph.num_edges));
    double initial_entropy = partition.getOverall_entropy();

    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        int num_batches = 10;
        int batch_size = int(ceil(graph.num_vertices / num_batches));
        for (int batch = 0; batch < graph.num_vertices / batch_size; ++batch) {
            int start = batch * batch_size;
            int end = std::min(graph.num_vertices, (batch + 1) * batch_size);
            // Block assignment used to re-create the Partition after each batch to improve mixing time of
            // asynchronous Gibbs sampling
            std::vector<int> block_assignment(partition.getBlock_assignment());
            #pragma omp parallel for
            for (int vertex = start; vertex < end; ++vertex) {
                VertexMove proposal = propose_gibbs_move(partition, vertex, graph.out_neighbors,
                                                         graph.in_neighbors);
                if (proposal.did_move) {
                    vertex_moves++;
                    delta_entropy += proposal.delta_entropy;
                    block_assignment[vertex] = proposal.proposed_block;
                }
            }
            partition = Partition(partition.getNum_blocks(), graph.out_neighbors,
                                  partition.getBlock_reduction_rate(), block_assignment);
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, partitions, initial_entropy, delta_entropies)) {
            break;
        }
    }
    partition.setOverall_entropy(overall_entropy(partition, graph.num_vertices, graph.num_edges));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << partition.getOverall_entropy() << std::endl;
    return partition;
}
