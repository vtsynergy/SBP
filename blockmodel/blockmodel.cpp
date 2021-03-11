#include "blockmodel.hpp"

std::vector<int> Blockmodel::build_mapping(std::vector<int> &values) {
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
    std::vector<int> mapping = build_mapping(this->block_assignment);
    for (int i = 0; i < this->block_assignment.size(); ++i) {
        int block = this->block_assignment[i];
        int new_block = mapping[block];
        this->block_assignment[i] = new_block;
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
    blockmodel_copy.block_assignment = std::vector<int>(this->block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy.blockmodel = this->blockmodel.copy();
    blockmodel_copy.block_degrees = std::vector<int>(this->block_degrees);
    blockmodel_copy.block_degrees_out = std::vector<int>(this->block_degrees_out);
    blockmodel_copy.block_degrees_in = std::vector<int>(this->block_degrees_in);
    blockmodel_copy._block_sizes = std::vector<int>(this->_block_sizes);
    // Create a Sampler as well?
    blockmodel_copy.num_blocks_to_merge = 0;
    return blockmodel_copy;
}

Blockmodel Blockmodel::from_sample(int num_blocks, NeighborList &neighbors, std::vector<int> &sample_block_membership,
                                 std::map<int, int> &mapping, float block_reduction_rate) {
    // Fill in initial block assignment
    std::vector<int> block_assignment = utils::constant<int>(neighbors.size(), -1);
    for (const auto &item : mapping) {
        block_assignment[item.first] = sample_block_membership[item.second];
    }
    // Every unassigned block gets assigned to the next block number
    int next_block = num_blocks;
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        if (block_assignment[vertex] >= 0) {
            continue;
        }
        block_assignment[vertex] = next_block;
        next_block++;
    }
    // Every previously unassigned block gets assigned to the block it's most connected to
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        if (block_assignment[vertex] < num_blocks) {
            continue;
        }
        std::vector<int> block_counts = utils::constant<int>(num_blocks, 0);
        // TODO: this can only handle unweighted graphs
        std::vector<int> vertex_neighbors = neighbors[vertex];
        for (uint i = 0; i < vertex_neighbors.size(); ++i) {
            int neighbor = vertex_neighbors[i];
            int neighbor_block = block_assignment[neighbor];
            if (neighbor_block < num_blocks) {
                block_counts[neighbor_block]++;
            }
        }
        int new_block = utils::argmax<int>(block_counts);
        // block_counts.maxCoeff(&new_block);
        block_assignment[vertex] = new_block;
    }
    return Blockmodel(num_blocks, neighbors, block_reduction_rate, block_assignment);
}

void Blockmodel::initialize_edge_counts(NeighborList &neighbors) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    this->blockmodel = DictTransposeMatrix(this->num_blocks, this->num_blocks);
    this->sampler = Sampler(this->num_blocks);
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
        int block = this->block_assignment[vertex];
        for (int i = 0; i < vertex_neighbors.size(); ++i) {
            // Get count
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->block_assignment[neighbor];
            // TODO: change this once code is updated to support weighted graphs
            int weight = 1;
            // int weight = vertex_neighbors[i];
            // Update blockmodel
            this->blockmodel.add(block, neighbor_block, weight);
            // Update degrees
            this->block_degrees_out[block] += weight;
            this->block_degrees_in[neighbor_block] += weight;
            this->sampler.insert(block, neighbor_block);
        }
        this->_block_sizes[block]++;
    }
    // Count block degrees
    this->block_degrees = this->block_degrees_out + this->block_degrees_in;
}

double Blockmodel::log_posterior_probability() {
    Indices nonzero_indices = this->blockmodel.nonzero();
    std::vector<double> values = utils::to_double<int>(this->blockmodel.values());
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

void Blockmodel::merge_blocks(int from_block, int to_block) {
    for (int index = 0; index < this->block_assignment.size(); ++index) {
        if (this->block_assignment[index] == from_block) {
            this->block_assignment[index] = to_block;
            this->_block_sizes[index]--;
            this->_block_sizes[to_block]++;
        }
    }
}

void Blockmodel::move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                            std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                            std::vector<int> &new_block_degrees) {
    this->block_assignment[vertex] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->block_degrees_out = new_block_degrees_out;
    this->block_degrees_in = new_block_degrees_in;
    this->block_degrees = new_block_degrees;
    this->_block_sizes[current_block]--;
    this->_block_sizes[new_block]++;
}

int Blockmodel::sample(int block) {
    return this->sampler.sample(block);
}

void Blockmodel::set_block_membership(int vertex, int block) {
    int current_block = this->block_assignment[vertex];
    this->block_assignment[vertex] = block;
    if (current_block >= 0)
        this->_block_sizes[current_block]--;
    this->_block_sizes[block]++;
}

void Blockmodel::update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates) {
    this->blockmodel.update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                        updates.block_col, updates.proposal_col);
}

void Sampler::insert(int from, int to) {
    if (from == to) return;
    this->neighbors[from].insert(to);
    this->neighbors[to].insert(from);
}

int Sampler::sample(int block) {
    const std::set<int> &neighborhood = this->neighbors[block];
    if (neighborhood.empty()) {  // sample a random block
        std::uniform_int_distribution<int> distribution(0, this->num_blocks - 2);
        int sampled = distribution(generator);
        if (sampled >= block) {
            sampled++;
        }
        return sampled;
    }
    std::uniform_int_distribution<int> distribution(0, neighborhood.size() - 1);
    int index = distribution(generator);
    // std::set doesn't have access by index - use iterator instead.
    std::set<int>::iterator it = neighborhood.begin();
    std::advance(it, index);
    int sampled = *it;
    return sampled;
}
