#include "dist_blockmodel.hpp"

std::vector<int> DistBlockmodel::build_mapping(std::vector<int> &values) {
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

// TODO: move to block_merge.cpp
void DistBlockmodel::carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                                       const std::vector<int> &best_merge_for_each_block) {
    std::vector<int> best_merges = sort_indices(delta_entropy_for_each_block);
    std::vector<int> block_map = utils::range<int>(0, this->_num_blocks);
    int num_merged = 0;
    int counter = 0;
    while (num_merged < this->_num_blocks_to_merge) {
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
    std::vector<int> mapping = build_mapping(this->_assignment);
    for (int i = 0; i < this->_assignment.size(); ++i) {
        int block = this->_assignment[i];
        int new_block = mapping[block];
        this->_assignment[i] = new_block;
    }
    this->_num_blocks -= this->_num_blocks_to_merge;
}

// DistBlockmodel DistBlockmodel::clone_with_true_block_membership(NeighborList &neighbors,
//                                                                 std::vector<int> &true_block_membership) {
//     int num_blocks = 0;
//     std::vector<int> uniques = utils::constant<int>(true_block_membership.size(), 0);
//     for (uint i = 0; i < true_block_membership.size(); ++i) {
//         int membership = true_block_membership[i];
//         uniques[membership] = 1; // mark as used
//     }
//     for (uint block = 0; block < uniques.size(); ++block) {
//         if (uniques[block] == 1) {
//             num_blocks++;
//         }
//     }
//     return DistBlockmodel(num_blocks, neighbors, true_block_membership);
// }

DistBlockmodel DistBlockmodel::copy() {
    DistBlockmodel blockmodel_copy = DistBlockmodel();
    blockmodel_copy._num_blocks = this->_num_blocks;
    blockmodel_copy._global_num_blocks = this->_global_num_blocks;
    blockmodel_copy._assignment = std::vector<int>(this->_assignment);
    blockmodel_copy._overall_entropy = this->_overall_entropy;
    blockmodel_copy._blockmatrix = this->_blockmatrix->copyDistSparseMatrix();
    blockmodel_copy._degrees = std::vector<int>(this->_degrees);
    blockmodel_copy._degrees_out = std::vector<int>(this->_degrees_out);
    blockmodel_copy._degrees_in = std::vector<int>(this->_degrees_in);
    blockmodel_copy._num_blocks_to_merge = 0;
    blockmodel_copy.empty = false;
    return blockmodel_copy;
}

// DistBlockmodel DistBlockmodel::from_sample(int num_blocks, NeighborList &neighbors, std::vector<int> &sample_block_membership,
//                                  std::map<int, int> &mapping, float block_reduction_rate) {
//     // Fill in initial block assignment
//     std::vector<int> block_assignment = utils::constant<int>(neighbors.size(), -1);
//     for (const auto &item : mapping) {
//         block_assignment[item.first] = sample_block_membership[item.second];
//     }
//     // Every unassigned block gets assigned to the next block number
//     int next_block = num_blocks;
//     for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
//         if (block_assignment[vertex] >= 0) {
//             continue;
//         }
//         block_assignment[vertex] = next_block;
//         next_block++;
//     }
//     // Every previously unassigned block gets assigned to the block it's most connected to
//     for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
//         if (block_assignment[vertex] < num_blocks) {
//             continue;
//         }
//         std::vector<int> block_counts = utils::constant<int>(num_blocks, 0);
//         // TODO: this can only handle unweighted graphs
//         std::vector<int> vertex_neighbors = neighbors[vertex];
//         for (uint i = 0; i < vertex_neighbors.size(); ++i) {
//             int neighbor = vertex_neighbors[i];
//             int neighbor_block = block_assignment[neighbor];
//             if (neighbor_block < num_blocks) {
//                 block_counts[neighbor_block]++;
//             }
//         }
//         int new_block = utils::argmax<int>(block_counts);
//         // block_counts.maxCoeff(&new_block);
//         block_assignment[vertex] = new_block;
//     }
//     return DistBlockmodel(num_blocks, neighbors, block_reduction_rate, block_assignment);
// }

// void DistBlockmodel::initialize_edge_counts(const NeighborList &neighbors, const MPI &mpi, const std::vector<int> &myblocks) {
void DistBlockmodel::initialize_edge_counts(const NeighborList &neighbors, const std::vector<int> &myblocks) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    // this->_blockmatrix = new DistDictMatrix(this->_global_num_blocks, this->_global_num_blocks, mpi, myblocks);
    this->_blockmatrix = new DistDictMatrix(this->_global_num_blocks, this->_global_num_blocks, myblocks);
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    this->_degrees_in = utils::constant<int>(this->_num_blocks, 0);
    this->_degrees_out = utils::constant<int>(this->_num_blocks, 0);
    // Initialize the blockmodel
    // TODO: find a way to parallelize the matrix filling step
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.size() == 0) {
            continue;
        }
        int block = this->_assignment[vertex];
        for (int i = 0; i < vertex_neighbors.size(); ++i) {
            // Get count
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_assignment[neighbor];
            // TODO: change this once code is updated to support weighted graphs
            int weight = 1;
            // int weight = vertex_neighbors[i];
            // Update blockmodel
            if (this->_blockmatrix->owns(block))
                this->_blockmatrix->add(block, neighbor_block, weight);
            // Update degrees
            this->_degrees_out[block] += weight;
            this->_degrees_in[neighbor_block] += weight;
        }
    }
    // Count block degrees
    this->_degrees = this->_degrees_out + this->_degrees_in;
    exit(-10);
}

double DistBlockmodel::log_posterior_probability() {
    Indices nonzero_indices = this->_blockmatrix->nonzero();
    std::vector<double> values = utils::to_double<int>(this->_blockmatrix->values());
    std::vector<double> degrees_in;
    std::vector<double> degrees_out;
    for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
        degrees_in.push_back(this->_degrees_in[nonzero_indices.cols[i]]);
        degrees_out.push_back(this->_degrees_out[nonzero_indices.rows[i]]);
    }
    std::vector<double> temp = values * utils::nat_log<double>(
        values / (degrees_out * degrees_in));
    return utils::sum<double>(temp);
}

void DistBlockmodel::merge_blocks(int from_block, int to_block) {
    for (int index = 0; index < this->_assignment.size(); ++index) {
        if (this->_assignment[index] == from_block) {
            this->_assignment[index] = to_block;
        }
    }
};

void DistBlockmodel::move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                            std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                            std::vector<int> &new_block_degrees) {
    this->_assignment[vertex] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->_degrees_out = new_block_degrees_out;
    this->_degrees_in = new_block_degrees_in;
    this->_degrees = new_block_degrees;
};

void DistBlockmodel::set_block_membership(int vertex, int block) { this->_assignment[vertex] = block; }

std::vector<int> DistBlockmodel::sort_indices(const std::vector<double> &unsorted) {
    // initialize original index locations
    std::vector<int> indices = utils::range<int>(0, unsorted.size());

    // sort indexes based on comparing values in unsorted
    std::sort(indices.data(), indices.data() + indices.size(),
              [unsorted](size_t i1, size_t i2) { return unsorted[i1] < unsorted[i2]; });

    return indices;
}

void DistBlockmodel::update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates) {
    this->_blockmatrix->update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                         updates.block_col, updates.proposal_col);
}

TwoHopBlockmodel TwoHopBlockmodel::copy() {
    TwoHopBlockmodel blockmodel_copy = TwoHopBlockmodel(this->num_blocks, this->block_reduction_rate);
    blockmodel_copy._block_assignment = std::vector<int>(this->_block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy._blockmatrix = this->_blockmatrix->copy();
    blockmodel_copy.block_degrees = std::vector<int>(this->block_degrees);
    blockmodel_copy.block_degrees_out = std::vector<int>(this->block_degrees_out);
    blockmodel_copy.block_degrees_in = std::vector<int>(this->block_degrees_in);
    blockmodel_copy.num_blocks_to_merge = 0;
    return blockmodel_copy;
}

void TwoHopBlockmodel::initialize_edge_counts(const NeighborList &neighbors) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    if (args.transpose) {
        this->_blockmatrix = new DictTransposeMatrix(this->num_blocks, this->num_blocks);
    } else {
        this->_blockmatrix = new DictMatrix(this->num_blocks, this->num_blocks);
    }
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    this->block_degrees_in = utils::constant<int>(this->num_blocks, 0);
    this->block_degrees_out = utils::constant<int>(this->num_blocks, 0);
    // First pass: find out which blocks are in the 2-hop radius of my blocks
    // TODO: make this a class variable for testing; at runtime, can check if there is a missing block
    // I think there will be a missing block in mcmc phase vertex->neighbor->block->neighbor_block
    std::vector<bool> in_two_hop_radius = utils::constant<bool>(this->num_blocks, false);
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.size() == 0) {
            continue;
        }
        int block = this->_block_assignment[vertex];
        for (int i = 0; i < vertex_neighbors.size(); ++i) {
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_block_assignment[neighbor];
            if ((block % mpi.num_processes == mpi.rank) || (neighbor_block % mpi.num_processes == mpi.rank)) {
                in_two_hop_radius[block] = true;
                in_two_hop_radius[neighbor_block] = true;
            }
        }
    }
    int two_hop_radius_size = 0;
    for (const bool val : in_two_hop_radius) {
        if (val == true) two_hop_radius_size++;
    }
    std::cout << "rank " << mpi.rank << " : num blocks in 2-hop radius == " << two_hop_radius_size << std::endl;
    // Second pass: initialize the blockmodel
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.size() == 0) {
            continue;
        }
        int block = this->_block_assignment[vertex];
        if (in_two_hop_radius[block] == false) {
            continue;
        }
        for (int i = 0; i < vertex_neighbors.size(); ++i) {
            // Get count
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_block_assignment[neighbor];
            if (in_two_hop_radius[neighbor_block] == false) {
                continue;
            }
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
    // exit(-1000000);
}
