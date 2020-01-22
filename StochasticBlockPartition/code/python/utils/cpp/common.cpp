#include "common.hpp"

int common::choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights) {
    std::discrete_distribution<int> distribution(neighbor_weights.begin(), neighbor_weights.end());
    int index = distribution(generator);
    return neighbor_indices[index];
}

int common::choose_neighbor(Eigen::SparseVector<double> &multinomial_distribution) {
    std::discrete_distribution<int> distribution(
        multinomial_distribution.valuePtr(), multinomial_distribution.valuePtr() + multinomial_distribution.nonZeros());
    int index = distribution(generator);
    return multinomial_distribution.innerIndexPtr()[index];
}

common::NewBlockDegrees common::compute_new_block_degrees(int current_block, Partition &partition,
                                                          common::ProposalAndEdgeCounts &proposal) {
    Vector new_block_degrees_out = Vector(partition.getBlock_degrees_out());
    Vector new_block_degrees_in = Vector(partition.getBlock_degrees_in());
    Vector new_block_degrees_total = Vector(partition.getBlock_degrees());
    new_block_degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    new_block_degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    new_block_degrees_in[current_block] -= proposal.num_in_neighbor_edges;
    new_block_degrees_in[proposal.proposal] += proposal.num_in_neighbor_edges;
    new_block_degrees_total[current_block] -= proposal.num_neighbor_edges;
    new_block_degrees_total[proposal.proposal] += proposal.num_neighbor_edges;
    return NewBlockDegrees{new_block_degrees_out, new_block_degrees_in, new_block_degrees_total};
}

double common::delta_entropy_temp(Vector &row_or_col, Vector &block_degrees, int degree) {
    Eigen::ArrayXd row_or_col_double = row_or_col.cast<double>().array();
    Eigen::ArrayXd block_degrees_double = block_degrees.cast<double>().array();
    Eigen::ArrayXd result = row_or_col_double / block_degrees_double / degree;
    result = row_or_col_double * result.log();
    return result.sum();
}

Vector common::exclude_indices(Vector &in, int index1, int index2) {
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

Vector common::index_nonzero(Vector &values, Vector &indices) {
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

Vector common::nonzeros(Vector &in) {
    std::vector<int> values;
    for (Eigen::Index i = 0; i < in.size(); ++i) {
        int value = in[i];
        if (value != 0) {
            values.push_back(value);
        }
    }
    return Eigen::Map<Vector>(values.data(), values.size());
}

common::ProposalAndEdgeCounts common::propose_new_block(int current_block, EdgeWeights &out_blocks,
                                                             EdgeWeights &in_blocks, Vector &block_partition,
                                                             Partition &partition, bool block_merge) {
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
    if (block_merge) {
        block_degrees[current_block] = 0.0;
        double total_degrees = block_degrees.sum();
        if (total_degrees == 0.0) { // Neighbor block has no neighbors, so propose a random block
            int proposal = propose_random_block(current_block, num_blocks);
            return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    Eigen::SparseVector<double> block_degrees_sparse = block_degrees.sparseView();
    Eigen::SparseVector<double> multinomial_distribution = block_degrees_sparse / block_degrees.sum();
    int proposal = choose_neighbor(multinomial_distribution);
    return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

int common::propose_random_block(int current_block, int num_blocks) {
    std::uniform_int_distribution<int> distribution(0, num_blocks - 1);
    int proposed = distribution(generator);
    if (proposed >= current_block) {
        proposed++;
    }
    return proposed;
}
