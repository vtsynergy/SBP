#include "common.hpp"

int common::choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights) {
    std::discrete_distribution<int> distribution(neighbor_weights.begin(), neighbor_weights.end());
    int index = distribution(generator);
    return neighbor_indices[index];
}

int common::choose_neighbor(SparseVector<double> &multinomial_distribution) {
    // std::cout << "in choose_neighbor" << std::endl;
    // std::cout << "data.size = " << multinomial_distribution.data.size() << " idx.size = ";
    // std::cout << multinomial_distribution.idx.size() << std::endl;
    std::discrete_distribution<int> distribution(
        multinomial_distribution.data.begin(), multinomial_distribution.data.end());
    int index = distribution(generator);
    // std::cout << "index = " << index << std::endl;
    return multinomial_distribution.idx[index];
}

common::NewBlockDegrees common::compute_new_block_degrees(int current_block, Partition &partition,
                                                          common::ProposalAndEdgeCounts &proposal) {
    // TODO: These are copy constructors. Maybe getting rid of them will speed things up?
    std::vector<int> new_block_degrees_out(partition.getBlock_degrees_out());
    std::vector<int> new_block_degrees_in(partition.getBlock_degrees_in());
    std::vector<int> new_block_degrees_total(partition.getBlock_degrees());
    new_block_degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    new_block_degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    new_block_degrees_in[current_block] -= proposal.num_in_neighbor_edges;
    new_block_degrees_in[proposal.proposal] += proposal.num_in_neighbor_edges;
    new_block_degrees_total[current_block] -= proposal.num_neighbor_edges;
    new_block_degrees_total[proposal.proposal] += proposal.num_neighbor_edges;
    return NewBlockDegrees{new_block_degrees_out, new_block_degrees_in, new_block_degrees_total};
}

double common::delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree) {
    std::vector<double> row_or_col_double = utils::to_double<int>(row_or_col);
    std::vector<double> block_degrees_double = utils::to_double<int>(block_degrees);
    std::vector<double> result = row_or_col_double / block_degrees_double / degree;
    result = row_or_col_double * utils::nat_log<double>(result);
    return utils::sum<double>(result);
}

std::vector<int> common::exclude_indices(std::vector<int> &in, int index1, int index2) {
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

MapVector<int> common::exclude_indices(MapVector<int> &in, int index1, int index2) {
    MapVector<int> out;
    for (const std::pair<int, int> &element: in) {
        if (element.first == index1 || element.first == index2)
            continue;
        int offset = 0;
        if (element.first > index1)
            offset += 1;
        if (element.first > index2)
            offset += 1;
        out[element.first - offset] = element.second;
    }
    return out;
}

std::vector<int> common::index_nonzero(std::vector<int> &values, std::vector<int> &indices) {
    // if (omp_get_thread_num() == 0)
    //     std::cout << "dense version" << std::endl;
    std::vector<int> results;
    for (int i = 0; i < indices.size(); ++i) {
        int index = indices[i];
        if (index != 0) {
            int value = values[i];
            // if (omp_get_thread_num() == 0)
            //     std::cout << "pushing " << value << " because " << i << " = " << index << std::endl;
            results.push_back(value);
        }
    }
    return results;
}

std::vector<int> common::index_nonzero(std::vector<int> &values, MapVector<int> &indices_map) {
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

std::vector<int> common::nonzeros(std::vector<int> &in) {
    std::vector<int> values;
    for (int i = 0; i < in.size(); ++i) {
        int value = in[i];
        if (value != 0) {
            values.push_back(value);
        }
    }
    return values;
}

std::vector<int> common::nonzeros(MapVector<int> &in) {
    std::vector<int> values;
    for (const std::pair<int, int> &element : in) {
        if (element.second != 0) {
            values.push_back(element.second);
        }
    }
    return values;
}

common::ProposalAndEdgeCounts common::propose_new_block(int current_block, EdgeWeights &out_blocks,
                                                        EdgeWeights &in_blocks, std::vector<int> &block_partition,
                                                        Partition &partition, bool block_merge) {
    // std::cout << "in propose_new_block" << std::endl;
    std::vector<int> neighbor_indices = utils::concatenate<int>(out_blocks.indices, in_blocks.indices);
    std::vector<int> neighbor_weights = utils::concatenate<int>(out_blocks.values, in_blocks.values);
    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    int k = k_out + k_in;
    int num_blocks = partition.getNum_blocks();
    // std::cout << "done setting up" << std::endl;
    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        // std::cout << "choosing random block" << std::endl;
        int proposal = propose_random_block(current_block, num_blocks);
        // std::cout << "done choosing random block" << std::endl;
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    int neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
    int neighbor_block = block_partition[neighbor];
    // With a probability inversely proportional to block degree, propose a random block merge

    if (std::rand() <= (num_blocks / ((float)partition.getBlock_degrees()[neighbor_block] + num_blocks))) {
        // std::cout << "choosing random block 2" << std::endl;
        int proposal = propose_random_block(current_block, num_blocks);
        // std::cout << "done choosing random block 2" << std::endl;
        return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    // Build multinomial distribution
    std::vector<double> row = utils::to_double<int>(partition.getBlockmodel().getrow(neighbor_block));
    std::vector<double> col = utils::to_double<int>(partition.getBlockmodel().getcol(neighbor_block));
    std::vector<double> block_degrees = row + col;
    if (block_merge) {
        block_degrees[current_block] = 0.0;
        double total_degrees = utils::sum<double>(block_degrees);
        if (total_degrees == 0.0) { // Neighbor block has no neighbors, so propose a random block
            // std::cout << "choosing random block 3" << std::endl;
            int proposal = propose_random_block(current_block, num_blocks);
            // std::cout << "done choosing random block 3" << std::endl;
            return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    // std::cout << "choosing neighbor: sum of block degrees = " << utils::sum<double>(block_degrees) << std::endl;
    SparseVector<double> block_degrees_sparse = utils::to_sparse<double>(block_degrees);  // .sparseView();
    // std::cout << "converted to sparse with size: " << block_degrees_sparse.data.size() << std::endl;
    SparseVector<double> multinomial_distribution = block_degrees_sparse / utils::sum<double>(block_degrees);
    // std::cout << "division done with size: " << multinomial_distribution.data.size() << std::endl;
    int proposal = choose_neighbor(multinomial_distribution);
    // std::cout << "done choosing neighbor" << std::endl;
    return ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

int common::propose_random_block(int current_block, int num_blocks) {
    // Generate numbers 0..num_blocks-2 in order to exclude the current block
    std::uniform_int_distribution<int> distribution(0, num_blocks - 2);
    int proposed = distribution(generator);
    if (proposed >= current_block) {
        proposed++;
    }
    return proposed;
}
