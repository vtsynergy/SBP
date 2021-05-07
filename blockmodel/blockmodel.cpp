#include "blockmodel.hpp"

#include "assert.h"

#include "../args.hpp"

std::vector<int> Blockmodel::build_mapping(const std::vector<int> &values) const {
    std::map<int, bool> unique_map;
    for (int i = 0; i < values.size(); ++i) {
        unique_map[values[i]] = true;
    }
    std::vector<int> mapping = utils::constant<int>(values.size(), -1);
    int counter = 0;
    for (std::pair<int, bool> element : unique_map) {
        mapping[element.first] = counter;
        counter++;
    }
    return mapping;
}

std::vector<int> Blockmodel::sort_indices(const std::vector<double> &unsorted) {
    // initialize original index locations
    std::vector<int> indices = utils::range<int>(0, unsorted.size());

    // sort indexes based on comparing values in unsorted
    std::sort(indices.data(), indices.data() + indices.size(),
              [unsorted](size_t i1, size_t i2) { return unsorted[i1] < unsorted[i2]; });

    return indices;
}

// TODO: move to block_merge.cpp
void Blockmodel::carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                                       const std::vector<int> &best_merge_for_each_block) {
    std::vector<int> best_merges = sort_indices(delta_entropy_for_each_block);
    std::vector<int> block_map = utils::range<int>(0, this->num_blocks);
    int num_merged = 0;
    int counter = 0;
    while (num_merged < this->num_blocks_to_merge) {
        int merge_from = best_merges[counter];
        int merge_to = block_map[best_merge_for_each_block[merge_from]];
        counter++;
        if (merge_to != merge_from) {
            for (int i = 0; i < block_map.size(); ++i) {
                int block = block_map[i];
                if (block == merge_from) {
                    block_map[i] = merge_to;
                }
            }
            this->merge_blocks(merge_from, merge_to);
            num_merged++;
        }
    }
    std::vector<int> mapping = build_mapping(this->_block_assignment);
    for (int i = 0; i < this->_block_assignment.size(); ++i) {
        int block = this->_block_assignment[i];
        int new_block = mapping[block];
        this->_block_assignment[i] = new_block;
    }
    this->num_blocks -= this->num_blocks_to_merge;
}

Blockmodel Blockmodel::clone_with_true_block_membership(NeighborList &neighbors,
                                                        std::vector<int> &true_block_membership) {
    int num_blocks = 0;
    std::vector<int> uniques = utils::constant<int>(true_block_membership.size(), 0);
    for (uint i = 0; i < true_block_membership.size(); ++i) {
        int membership = true_block_membership[i];
        uniques[membership] = 1; // mark as used
    }
    for (uint block = 0; block < uniques.size(); ++block) {
        if (uniques[block] == 1) {
            num_blocks++;
        }
    }
    return Blockmodel(num_blocks, neighbors, this->block_reduction_rate, true_block_membership);
}

Blockmodel Blockmodel::copy() {
    Blockmodel blockmodel_copy = Blockmodel(this->num_blocks, this->block_reduction_rate);
    blockmodel_copy._block_assignment = std::vector<int>(this->_block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy._blockmatrix = this->_blockmatrix->copy();
    blockmodel_copy.block_degrees = std::vector<int>(this->block_degrees);
    blockmodel_copy.block_degrees_out = std::vector<int>(this->block_degrees_out);
    blockmodel_copy.block_degrees_in = std::vector<int>(this->block_degrees_in);
    blockmodel_copy.num_blocks_to_merge = 0;
    return blockmodel_copy;
}

Blockmodel Blockmodel::from_sample(int num_blocks, NeighborList &neighbors, std::vector<int> &sample_block_membership,
                                 std::map<int, int> &mapping, float block_reduction_rate) {
    // Fill in initial block assignment
    std::vector<int> _block_assignment = utils::constant<int>(neighbors.size(), -1);
    for (const auto &item : mapping) {
        _block_assignment[item.first] = sample_block_membership[item.second];
    }
    // Every unassigned block gets assigned to the next block number
    int next_block = num_blocks;
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        if (_block_assignment[vertex] >= 0) {
            continue;
        }
        _block_assignment[vertex] = next_block;
        next_block++;
    }
    // Every previously unassigned block gets assigned to the block it's most connected to
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        if (_block_assignment[vertex] < num_blocks) {
            continue;
        }
        std::vector<int> block_counts = utils::constant<int>(num_blocks, 0);
        // TODO: this can only handle unweighted graphs
        std::vector<int> vertex_neighbors = neighbors[vertex];
        for (uint i = 0; i < vertex_neighbors.size(); ++i) {
            int neighbor = vertex_neighbors[i];
            int neighbor_block = _block_assignment[neighbor];
            if (neighbor_block < num_blocks) {
                block_counts[neighbor_block]++;
            }
        }
        int new_block = utils::argmax<int>(block_counts);
        // block_counts.maxCoeff(&new_block);
        _block_assignment[vertex] = new_block;
    }
    return Blockmodel(num_blocks, neighbors, block_reduction_rate, _block_assignment);
}

void Blockmodel::initialize_edge_counts(const NeighborList &neighbors) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    if (args.transpose) {
        this->_blockmatrix = new DictTransposeMatrix(this->num_blocks, this->num_blocks);
    } else {
        this->_blockmatrix = new DictMatrix(this->num_blocks, this->num_blocks);
    }
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    this->block_degrees_in = utils::constant<int>(this->num_blocks, 0);
    this->block_degrees_out = utils::constant<int>(this->num_blocks, 0);
    // Initialize the blockmodel
    // TODO: find a way to parallelize the matrix filling step
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.size() == 0) {
            continue;
        }
        int block = this->_block_assignment[vertex];
        for (int i = 0; i < vertex_neighbors.size(); ++i) {
            // Get count
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_block_assignment[neighbor];
            // TODO: change this once code is updated to support weighted graphs
            int weight = 1;
            // int weight = vertex_neighbors[i];
            // Update blockmodel
            this->_blockmatrix->add(block, neighbor_block, weight);
            // Update degrees
            this->block_degrees_out[block] += weight;
            this->block_degrees_in[neighbor_block] += weight;
        }
    }
    // Count block degrees
    if (args.undirected) {
        this->block_degrees = std::vector<int>(this->block_degrees_out);
    } else {
        this->block_degrees = this->block_degrees_out + this->block_degrees_in; 
    }
}

double Blockmodel::log_posterior_probability() const {
    Indices nonzero_indices = this->_blockmatrix->nonzero();
    std::vector<double> values = utils::to_double<int>(this->_blockmatrix->values());
    std::vector<double> degrees_in;
    std::vector<double> degrees_out;
    for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
        degrees_in.push_back(this->block_degrees_in[nonzero_indices.cols[i]]);
        degrees_out.push_back(this->block_degrees_out[nonzero_indices.rows[i]]);
    }
    std::vector<double> temp = values * utils::nat_log<double>(
        values / (degrees_out * degrees_in));
    return utils::sum<double>(temp);
}

double Blockmodel::log_posterior_probability(int num_edges) const {
    if (args.undirected) {
        Indices nonzero_indices = this->_blockmatrix->nonzero();
        std::vector<double> values = utils::to_double<int>(this->_blockmatrix->values());
        std::vector<double> degrees_in;
        std::vector<double> degrees_out;
        for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
            // This is OK bcause block_degrees_in == block_degrees_out == block_degrees
            degrees_in.push_back(this->block_degrees_in[nonzero_indices.cols[i]] / (2.0));
            degrees_out.push_back(this->block_degrees_out[nonzero_indices.rows[i]] / (2.0));
        }
        std::vector<double> temp = values * utils::nat_log<double>(
            (values / (2.0)) / (degrees_out * degrees_in));
        double result = 0.5 * utils::sum<double>(temp);
        return result;
    }
    // if (args.undirected) {
    //     Indices nonzero_indices = this->_blockmatrix->nonzero();
    //     std::vector<double> values = utils::to_double<int>(this->_blockmatrix->values());
    //     std::vector<double> degrees_in;
    //     std::vector<double> degrees_out;
    //     for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
    //         // This is OK bcause block_degrees_in == block_degrees_out == block_degrees
    //         degrees_in.push_back(this->block_degrees_in[nonzero_indices.cols[i]] / (2.0 * num_edges));
    //         degrees_out.push_back(this->block_degrees_out[nonzero_indices.rows[i]] / (2.0 * num_edges));
    //     }
    //     std::vector<double> temp = values * utils::nat_log<double>(
    //         (values / (2.0 * num_edges)) / (degrees_out * degrees_in));
    //     // std::cout << "temp: ";
    //     // utils::print<double>(temp);
    //     std::cout << "for index 10: " << std::endl;
    //     std::cout << "num_edges = " << num_edges;
    //     std::cout << " values[10] = " << values[10];
    //     std::cout << " d_in[10] = " << block_degrees_in[nonzero_indices.cols[10]];
    //     std::cout << " d_out[10] = " << block_degrees_out[nonzero_indices.rows[10]];
    //     std::cout << " temp[10] = " << temp[10] << std::endl;
    //     double result = 0.5 * utils::sum<double>(temp);
    //     return result;
    // }
    return log_posterior_probability();
}

void Blockmodel::merge_blocks(int from_block, int to_block) {
    for (int index = 0; index < this->_block_assignment.size(); ++index) {
        if (this->_block_assignment[index] == from_block) {
            this->_block_assignment[index] = to_block;
        }
    }
};

void Blockmodel::move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                             std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                             std::vector<int> &new_block_degrees) {
    this->_block_assignment[vertex] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->block_degrees_out = new_block_degrees_out;
    this->block_degrees_in = new_block_degrees_in;
    this->block_degrees = new_block_degrees;
};

void Blockmodel::set_block_membership(int vertex, int block) { this->_block_assignment[vertex] = block; }

void Blockmodel::update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates) {
    this->_blockmatrix->update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                           updates.block_col, updates.proposal_col);
}
