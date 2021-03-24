#include "common.hpp"

namespace common {

int choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights) {
    std::discrete_distribution<int> distribution(neighbor_weights.begin(), neighbor_weights.end());
    int index = distribution(generator);
    return neighbor_indices[index];
}

int choose_neighbor_uniform(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights) {
    std::uniform_int_distribution<int> distribution(0, neighbor_indices.size() - 1);
    int index = distribution(generator);
    return neighbor_indices[index];
}

int choose_neighbor(const SparseVector<double> &multinomial_distribution) {
    // std::cout << "in choose_neighbor" << std::endl;
    // std::cout << "data.size = " << multinomial_distribution.data.size() << " idx.size = ";
    // std::cout << multinomial_distribution.idx.size() << std::endl;
    std::discrete_distribution<int> distribution(
        multinomial_distribution.data.begin(), multinomial_distribution.data.end());
    int index = distribution(generator);
    // std::cout << "index = " << index << std::endl;
    return multinomial_distribution.idx[index];
}

int choose_neighbor_uniform(const SparseVector<double> &multinomial_distribution) {
    std::uniform_int_distribution<int> distribution(0, multinomial_distribution.data.size() - 1);
    int index = distribution(generator);
    return multinomial_distribution.idx[index];
}

// TODO: do these calculations on the fly so you don't have to store 3 vectors of size B
NewBlockDegrees compute_new_block_degrees(int current_block, Blockmodel &blockmodel, ProposalAndEdgeCounts &proposal) {
    // TODO: These are copy constructors. Maybe getting rid of them will speed things up?
    std::vector<int> new_block_degrees_out(blockmodel.getBlock_degrees_out());
    std::vector<int> new_block_degrees_in(blockmodel.getBlock_degrees_in());
    std::vector<int> new_block_degrees_total(blockmodel.getBlock_degrees());
    new_block_degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    new_block_degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    new_block_degrees_in[current_block] -= proposal.num_in_neighbor_edges;
    new_block_degrees_in[proposal.proposal] += proposal.num_in_neighbor_edges;
    new_block_degrees_total[current_block] -= proposal.num_neighbor_edges;
    new_block_degrees_total[proposal.proposal] += proposal.num_neighbor_edges;
    return NewBlockDegrees{new_block_degrees_out, new_block_degrees_in, new_block_degrees_total};
}

double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree) {
    std::vector<double> row_or_col_double = utils::to_double<int>(row_or_col);
    std::vector<double> block_degrees_double = utils::to_double<int>(block_degrees);
    std::vector<double> result = row_or_col_double / block_degrees_double / degree;
    result = row_or_col_double * utils::nat_log<double>(result);
    return utils::sum<double>(result);
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree) {
    double result = 0.0;
    for (const std::pair<int, int> &pair : row_or_col) {
        if (pair.second == 0)  // 0s sometimes get inserted into the sparse matrix
            continue;
        double temp = (double) pair.second / (double) block_degrees[pair.first] / degree;
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    return result;
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal) {
    double result = 0.0;
    for (const std::pair<int, int> &pair : row_or_col) {
        // 0s sometimes get inserted into the sparse matrix
        if (pair.second == 0 || pair.first == current_block || pair.first == proposal)
            continue;
        double temp = (double) pair.second / (double) block_degrees[pair.first] / degree;
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    return result;
}

std::vector<int> exclude_indices(std::vector<int> &in, int index1, int index2) {
    std::vector<int> out = utils::constant<int>(in.size() - 1, 0);
    int count = 0;
    for (int i = 0; i < in.size(); ++i) {
        if (i == index1 || i == index2) {
            continue;
        }
        out[count] = in[i];
        count++;
    }
    return out;
}

MapVector<int>& exclude_indices(MapVector<int> &in, int index1, int index2) {
    // MapVector<int> out(in);
    in.erase(index1);
    in.erase(index2);
    return in;
}

std::vector<int> index_nonzero(std::vector<int> &values, std::vector<int> &indices) {
    std::vector<int> results;
    for (int i = 0; i < indices.size(); ++i) {
        int index = indices[i];
        if (index != 0) {
            int value = values[i];
            results.push_back(value);
        }
    }
    return results;
}

std::vector<int> index_nonzero(std::vector<int> &values, MapVector<int> &indices_map) {
    // if (omp_get_thread_num() == 0)
    //     std::cout << "sparse version" << std::endl;
    std::vector<int> results;
    for (const std::pair<int, int> &element : indices_map) {
        if (element.second != 0) {
            // if (omp_get_thread_num() == 0)
            //     std::cout << "pushing " << values[element.first] << " because " << element.first << " = " << element.second << std::endl;
            results.push_back(values[element.first]);
        }
    }
    return results;
}

std::vector<int> nonzeros(std::vector<int> &in) {
    std::vector<int> values;
    for (int i = 0; i < in.size(); ++i) {
        int value = in[i];
        if (value != 0) {
            values.push_back(value);
        }
    }
    return values;
}

std::vector<int> nonzeros(MapVector<int> &in) {
    std::vector<int> values;
    for (const std::pair<int, int> &element : in) {
        if (element.second != 0) {
            values.push_back(element.second);
        }
    }
    return values;
}

ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                        std::vector<int> &assignment, Blockmodel &blockmodel, bool block_merge) {
    // TODO: this results in neighbor_indices and neighbor_weights with repeats
    std::vector<int> neighbor_indices = utils::concatenate<int>(out_blocks.indices, in_blocks.indices);
    std::vector<int> neighbor_weights = utils::concatenate<int>(out_blocks.values, in_blocks.values);
    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    int k = k_out + k_in;
    int num_blocks = blockmodel.getNum_blocks();

    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        int proposal = propose_random_block(current_block, num_blocks);
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    int neighbor_block;
    if (block_merge) {
        neighbor_block = choose_neighbor(neighbor_indices, neighbor_weights);
    } else {
        int neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
        neighbor_block = assignment[neighbor];
    }

    // With a probability inversely proportional to block degree, propose a random block merge
    if (std::rand() <= (num_blocks / ((float)blockmodel.getBlock_degrees()[neighbor_block] + num_blocks))) {
        int proposal = propose_random_block(current_block, num_blocks);
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    // Build multinomial distribution
    double total_edges = 0.0;
    const DictTransposeMatrix &matrix = blockmodel.getBlockmodel();
    const MapVector<int> &col = matrix.getcol_sparse(neighbor_block);
    MapVector<int> edges = blockmodel.getBlockmodel().getrow_sparse(neighbor_block);
    for (const std::pair<int, int> &pair : col) {
        edges[pair.first] += pair.second;
    }
    if (block_merge) {  // Make sure proposal != current_block
        edges[current_block] = 0;
        total_edges = utils::sum<double, int>(edges);
        if (total_edges == 0.0) { // Neighbor block has no neighbors, so propose a random block
            int proposal = propose_random_block(current_block, num_blocks);
            return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    } else {
        total_edges = utils::sum<double, int>(edges);
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    SparseVector<double> multinomial_distribution;
    utils::div(edges, total_edges, multinomial_distribution);
    int proposal = choose_neighbor(multinomial_distribution);
    // int proposal = choose_neighbor_uniform(multinomial_distribution);
    // int proposal = blockmodel.sample(current_block);
    if (proposal == current_block) {
        std::cout << "proposal == current_block" << std::endl;
        proposal = propose_random_block(current_block, num_blocks);
    }
    return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

ProposalAndEdgeCounts propose_new_block(int current_block, Blockmodel &blockmodel, bool random) {
    int proposal;
    if (random)
        proposal = propose_random_block(current_block, blockmodel.getNum_blocks());
    else {
        int neighbor = blockmodel.sample(current_block, generator);
        if (neighbor == NULL_BLOCK)
            proposal = neighbor;
        else
            proposal = blockmodel.sample(neighbor, generator);
    }
    int kout = blockmodel.getBlock_degrees_out()[current_block];
    int kin = blockmodel.getBlock_degrees_in()[current_block];
    int k = kout + kin;
    return ProposalAndEdgeCounts{proposal, kout, kin, k};
}

ProposalAndEdgeCounts propose_new_block_mcmc(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                             std::vector<int> &assignment, Blockmodel &blockmodel, bool block_merge) {
    // TODO: this results in neighbor_indices and neighbor_weights with repeats
    std::vector<int> neighbor_indices = utils::concatenate<int>(out_blocks.indices, in_blocks.indices);
    std::vector<int> neighbor_weights = utils::concatenate<int>(out_blocks.values, in_blocks.values);
    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    int k = k_out + k_in;
    int num_blocks = blockmodel.getNum_blocks();

    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        int proposal = propose_random_block(current_block, num_blocks);
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    int neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
    int neighbor_block = assignment[neighbor];

    // // With a probability inversely proportional to block degree, propose a random block merge
    // if (std::rand() <= (num_blocks / ((float)blockmodel.getBlock_degrees()[neighbor_block] + num_blocks))) {
    //     int proposal = propose_random_block(current_block, num_blocks);
    //     return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    // }

    // Build multinomial distribution
    double total_edges = 0.0;
    const DictTransposeMatrix &matrix = blockmodel.getBlockmodel();
    const MapVector<int> &col = matrix.getcol_sparse(neighbor_block);
    MapVector<int> edges = blockmodel.getBlockmodel().getrow_sparse(neighbor_block);
    for (const std::pair<int, int> &pair : col) {
        edges[pair.first] += pair.second;
    }
    total_edges = utils::sum<double, int>(edges);
    // Propose a block based on the multinomial distribution of block neighbor edges
    SparseVector<double> multinomial_distribution;
    utils::div(edges, total_edges, multinomial_distribution);
    // int proposal = choose_neighbor(multinomial_distribution);
    int proposal = choose_neighbor_uniform(multinomial_distribution);
    // int proposal = blockmodel.sample(current_block);
    // if (proposal == current_block) {
        // std::cout << "proposal == current_block" << std::endl;
        // proposal = propose_random_block(current_block, num_blocks);
    // }
    return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

int propose_random_block(int current_block, int num_blocks) {
    // Generate numbers 0..num_blocks-2 in order to exclude the current block
    std::uniform_int_distribution<int> distribution(0, num_blocks - 2);
    int proposed = distribution(generator);
    if (proposed >= current_block) {
        proposed++;
    }
    return proposed;
}

} // common
