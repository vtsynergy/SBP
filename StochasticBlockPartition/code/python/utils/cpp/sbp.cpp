#include "sbp.hpp"

#include <math.h>

Partition& sbp::merge_blocks(Partition &partition, int num_agg_proposals_per_block,
                            std::vector<Matrix2Column> &out_neighbors) {
    // add block merge timings to evaluation
    int num_blocks = partition.getNum_blocks();
    Vector best_merge_for_each_block = Vector::Constant(num_blocks, -1);
    Eigen::VectorXd delta_entropy_for_each_block =
        Eigen::VectorXd::Constant(num_blocks, std::numeric_limits<double>::max());
    Vector block_partition = Vector::LinSpaced(num_blocks, 0, num_blocks - 1);

    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        for (int i = 0; i < num_agg_proposals_per_block; ++i) {
            ProposalEvaluation proposal = propose_merge(current_block, partition, block_partition);
            if (proposal.delta_entropy < delta_entropy_for_each_block[current_block]) {
                best_merge_for_each_block[current_block] = proposal.proposed_block;
                delta_entropy_for_each_block[current_block] = proposal.delta_entropy;
            }
        }
    }

    partition.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block);
    partition.initialize_edge_counts(out_neighbors);

    return partition;
}

sbp::ProposalEvaluation sbp::propose_merge(int current_block, Partition &partition, Vector &block_partition) {
    EdgeWeights out_blocks = partition.getBlockmodel().outgoing_edges(current_block);
    EdgeWeights in_blocks = partition.getBlockmodel().incoming_edges(current_block);
    ProposalAndEdgeCounts proposal =
        propose_new_block(current_block, out_blocks, in_blocks, block_partition, partition);
    BlockMergeEdgeCountUpdates updates = block_merge_edge_count_updates(
        partition.getBlockmodel(), current_block, proposal.proposal, out_blocks, in_blocks);
    NewBlockDegrees new_block_degrees = compute_new_block_degrees(current_block, partition, proposal);
    double delta_entropy = compute_delta_entropy(
        current_block, proposal.proposal, partition, updates, new_block_degrees);
    return ProposalEvaluation { proposal.proposal, delta_entropy };
}

Vector exclude_indices(Vector &in, int index1, int index2) {
    Vector out = Vector::Zero(in.size() - 2);
    Eigen::Index count = 0;
    for (Eigen::Index i = 0; i < in.size(); ++i) {
        if (i == index1 || i == index2) {
            continue;
        }
        out[count] = in[i];
        count++;
    }
    return out;
}

Vector index_nonzero(Vector &values, Vector &indices) {
    std::vector<int> results;
    for (Eigen::Index i = 0; i < indices.size(); ++i) {
        int index = indices[i];
        if (index != 0) {
            int value = values[i];
            results.push_back(value);
        }
    }
    return Eigen::Map<Vector>(results.data(), results.size());
}

Vector nonzeros(Vector &in) {
    std::vector<int> values;
    for (Eigen::Index i = 0; i < in.size(); ++i) {
        int value = in[i];
        if (value != 0) {
            values.push_back(value);
        }
    }
    return Eigen::Map<Vector>(values.data(), values.size());
}

double delta_entropy_temp(Vector &row_or_col, Vector &block_degrees, int degree) {
    Eigen::ArrayXd row_or_col_double = row_or_col.cast<double>().array();
    Eigen::ArrayXd block_degrees_double = block_degrees.cast<double>().array();
    Eigen::ArrayXd result = row_or_col_double / block_degrees_double / degree;
    result = row_or_col_double * result.log();
    return result.sum();
}

double sbp::compute_delta_entropy(int current_block, int proposal, Partition &partition,
                                  BlockMergeEdgeCountUpdates &updates, NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    Vector old_block_row = partition.getBlockmodel().getrow(current_block);  // M_r_t1
    Vector old_proposal_row = partition.getBlockmodel().getrow(proposal);  // M_s_t1
    Vector old_block_col = partition.getBlockmodel().getcol(current_block);  // M_t2_r
    Vector old_proposal_col = partition.getBlockmodel().getcol(proposal);  // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    Vector new_proposal_col = exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = exclude_indices(old_block_col, current_block, proposal);  // M_t2_r
    old_proposal_col = exclude_indices(old_proposal_col, current_block, proposal);  // M_t2_s
    Vector new_block_degrees_out = exclude_indices(block_degrees.block_degrees_out, current_block, proposal);
    Vector old_block_degrees_out = exclude_indices(partition.getBlock_degrees_out(), current_block, proposal);

    // Remove 0 indices
    Vector new_proposal_row_degrees_in = index_nonzero(block_degrees.block_degrees_in, updates.proposal_row);
    Vector new_proposal_row = nonzeros(updates.proposal_row);
    Vector new_proposal_col_degrees_out = index_nonzero(new_block_degrees_out, new_proposal_col);
    new_proposal_col = nonzeros(new_proposal_col);

    Vector old_block_row_degrees_in = index_nonzero(partition.getBlock_degrees_in(), old_block_row);
    Vector old_proposal_row_degrees_in = index_nonzero(partition.getBlock_degrees_in(), old_proposal_row);
    old_block_row = nonzeros(old_block_row);
    old_proposal_row = nonzeros(old_proposal_row);
    Vector old_block_col_degrees_out = index_nonzero(old_block_degrees_out, old_block_col);
    Vector old_proposal_col_degrees_out = index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = nonzeros(old_block_col);
    old_proposal_col = nonzeros(old_proposal_col);
    
    double delta_entropy = 0.0;
    delta_entropy -= delta_entropy_temp(
        new_proposal_row, new_proposal_row_degrees_in, block_degrees.block_degrees_out[proposal]);
    delta_entropy -= delta_entropy_temp(
        new_proposal_col, new_proposal_col_degrees_out, block_degrees.block_degrees_in[proposal]);
    delta_entropy += delta_entropy_temp(
        old_block_row, old_block_row_degrees_in, partition.getBlock_degrees_out()[current_block]);
    delta_entropy += delta_entropy_temp(
        old_proposal_row, old_proposal_row_degrees_in, partition.getBlock_degrees_out()[proposal]);
    delta_entropy += delta_entropy_temp(
        old_block_col, old_block_col_degrees_out, partition.getBlock_degrees_in()[current_block]);
    delta_entropy += delta_entropy_temp(
        old_proposal_col, old_proposal_col_degrees_out, partition.getBlock_degrees_in()[proposal]);
    return delta_entropy;
}

sbp::ProposalAndEdgeCounts sbp::propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                                  Vector &block_partition, Partition &partition) {
    std::vector<int> neighbor_indices = util::concatenate<int>(out_blocks.indices, in_blocks.indices);
    std::vector<int> neighbor_weights = util::concatenate<int>(out_blocks.values, in_blocks.values);
    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    int k = k_out + k_in;
    int num_blocks = partition.getNum_blocks();
    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        int proposal = propose_random_block(current_block, num_blocks);
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    int neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
    int neighbor_block = block_partition[neighbor];
    // With a probability inversely proportional to block degree, propose a random block merge

    if (std::rand() <= (num_blocks / ((float)partition.getBlock_degrees()[neighbor_block] + num_blocks))) {
        int proposal = propose_random_block(current_block, num_blocks);
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    // Build multinomial distribution
    Eigen::VectorXd row = partition.getBlockmodel().getrow(neighbor_block).cast<double>();
    Eigen::VectorXd col = partition.getBlockmodel().getcol(neighbor_block).cast<double>();
    Eigen::VectorXd block_degrees = row + col;
    block_degrees[current_block] = 0.0;
    double total_degrees = block_degrees.sum();
    if (total_degrees == 0.0) { // Neighbor block has no neighbors, so propose a random block
        int proposal = propose_random_block(current_block, num_blocks);
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    Eigen::SparseVector<double> block_degrees_sparse = block_degrees.sparseView();
    Eigen::SparseVector<double> multinomial_distribution = block_degrees_sparse / total_degrees;
    int proposal = choose_neighbor(multinomial_distribution);
    return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

sbp::BlockMergeEdgeCountUpdates sbp::block_merge_edge_count_updates(BoostMappedMatrix &blockmodel, int current_block,
                                                                    int proposed_block, EdgeWeights &out_blocks,
                                                                    EdgeWeights &in_blocks) {
    Vector proposal_row = blockmodel.getrow(proposed_block);
    Vector proposal_col = blockmodel.getcol(proposed_block);
    int count_self = blockmodel.get(current_block, current_block);
    int count_in = count_self, count_out = count_self;
    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int index = in_blocks.indices[i];
        int value = in_blocks.values[i];
        if (index == proposed_block) {
            count_in += value;
        }
        proposal_col[index] += value;
    }
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int index = out_blocks.indices[i];
        int value = out_blocks.values[i];
        if (index == proposed_block) {
            count_out += value;
        }
        proposal_row[index] += value;
    }
    proposal_row[current_block] -= count_in;
    proposal_row[proposed_block] += count_in;
    proposal_col[current_block] -= count_out;
    proposal_col[proposed_block] += count_out;
    return BlockMergeEdgeCountUpdates {proposal_row, proposal_col};
}

sbp::NewBlockDegrees sbp::compute_new_block_degrees(int current_block, Partition &partition,
                                                    ProposalAndEdgeCounts &proposal) {
    Vector new_block_degrees_out = Vector(partition.getBlock_degrees_out());
    Vector new_block_degrees_in = Vector(partition.getBlock_degrees_in());
    Vector new_block_degrees_total = Vector(partition.getBlock_degrees());
    new_block_degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    new_block_degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    new_block_degrees_in[current_block] -= proposal.num_in_neighbor_edges;
    new_block_degrees_in[proposal.proposal] += proposal.num_in_neighbor_edges;
    new_block_degrees_total[current_block] -= proposal.num_neighbor_edges;
    new_block_degrees_total[proposal.proposal] += proposal.num_neighbor_edges;
    return NewBlockDegrees {new_block_degrees_out, new_block_degrees_in, new_block_degrees_total};
}

int sbp::propose_random_block(int current_block, int num_blocks) {
    std::uniform_int_distribution<int> distribution(0, num_blocks - 1);
    int proposed = distribution(generator);
    if (proposed >= current_block) {
        proposed++;
    }
    return proposed;
}

int sbp::choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights) {
    std::discrete_distribution<int> distribution(neighbor_weights.begin(), neighbor_weights.end());
    int index = distribution(generator);
    return neighbor_indices[index];
}

int sbp::choose_neighbor(Eigen::SparseVector<double> &multinomial_distribution) {
    std::discrete_distribution<int> distribution(multinomial_distribution.valuePtr(),
                                                 multinomial_distribution.valuePtr() + multinomial_distribution.nonZeros());
    int index = distribution(generator);
    return multinomial_distribution.innerIndexPtr()[index];
}
