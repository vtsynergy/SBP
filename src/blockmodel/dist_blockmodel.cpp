#include "dist_blockmodel.hpp"

#include <unordered_set>

void TwoHopBlockmodel::build_two_hop_blockmodel(const NeighborList &neighbors) {
    if (args.distribute == "none") {
        this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
        return;
    }
    if (args.distribute == "2hop-snowball") {
        this->_my_blocks = std::vector<bool>(this->num_blocks, false);
        for (int v = 0; v < (int) neighbors.size(); ++v) {
            if (this->owns_vertex(v)) {
                int b = this->block_assignment(v);
                this->_my_blocks[b] = true;
            }
        }
    }
    // I think there will be a missing block in mcmc phase vertex->neighbor->block->neighbor_block
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, false);
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.empty()) {
            continue;
        }
        int block = this->_block_assignment[vertex];
        for (size_t i = 0; i < vertex_neighbors.size(); ++i) {
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_block_assignment[neighbor];
            if (this->_my_blocks[block] || this->_my_blocks[neighbor_block]) {
            // if ((block % mpi.num_processes == mpi.rank) || (neighbor_block % mpi.num_processes == mpi.rank)) {
                this->_in_two_hop_radius[block] = true;
                this->_in_two_hop_radius[neighbor_block] = true;
            }
        }
    }
    int two_hop_radius_size = 0;
    for (const bool val : this->_in_two_hop_radius) {
        if (val) two_hop_radius_size++;
    }
    if (mpi.rank == 0) std::cout << "rank 0: num blocks in 2-hop radius == " << two_hop_radius_size << " / " << this->num_blocks << std::endl;
}

TwoHopBlockmodel TwoHopBlockmodel::copy() {
    TwoHopBlockmodel blockmodel_copy = TwoHopBlockmodel(this->num_blocks, this->block_reduction_rate);
    blockmodel_copy._block_assignment = std::vector<int>(this->_block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy._blockmatrix = std::shared_ptr<ISparseMatrix>(this->_blockmatrix->copy());
    blockmodel_copy._block_degrees = std::vector<int>(this->_block_degrees);
    blockmodel_copy._block_degrees_out = std::vector<int>(this->_block_degrees_out);
    blockmodel_copy._block_degrees_in = std::vector<int>(this->_block_degrees_in);
    blockmodel_copy._in_two_hop_radius = std::vector<bool>(this->_in_two_hop_radius);
    blockmodel_copy.num_blocks_to_merge = 0;
    blockmodel_copy._my_blocks = std::vector<bool>(this->_my_blocks);
    blockmodel_copy.empty = this->empty;
    return blockmodel_copy;
}

void TwoHopBlockmodel::distribute(const Graph &graph) {
    if (args.distribute == "none")
        distribute_none();
    else if (args.distribute == "2hop-round-robin")
        distribute_2hop_round_robin(graph.out_neighbors());
    else if (args.distribute == "2hop-size-balanced")
        distribute_2hop_size_balanced(graph.out_neighbors());
    else if (args.distribute == "2hop-snowball")
        distribute_2hop_snowball(graph.out_neighbors());
    else if (args.distribute == "2hop-super-snowball")
        distribute_2hop_super_snowball(graph.out_neighbors(), graph.in_neighbors());
    else
        distribute_2hop_snowball(graph.out_neighbors());
}

void TwoHopBlockmodel::distribute_none() {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    for (int i = mpi.rank; i < this->num_blocks; i += mpi.num_processes)
        this->_my_blocks[i] = true;
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_2hop_round_robin(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    for (int i = mpi.rank; i < this->num_blocks; i += mpi.num_processes)
        this->_my_blocks[i] = true;
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::distribute_2hop_size_balanced(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    std::vector<std::pair<int,int>> block_sizes = this->sorted_block_sizes();
    for (int i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    for (int i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::distribute_2hop_snowball(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    // std::cout << "my vertices size: " << this->_my_vertices.size() << " neighbors size: " << neighbors.size() << std::endl;
    if (this->_my_vertices.size() == neighbors.size()) {  // if already done sampling, no need to do it again
        std::cout << "already done sampling, now just re-assigning blocks based on sampled vertices" << std::endl;
        for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
            if (this->_my_vertices[vertex] == 0) continue;
            int block = this->_block_assignment[vertex];
            this->_my_blocks[block] = true;
        }
    } else {
        int target = ceil((double) neighbors.size() / (double) mpi.num_processes);
        this->_my_vertices = utils::constant<int>(neighbors.size(), 0);  // cannot send vector<bool>.data() over MPI
        std::unordered_set<int> frontier;
        // Snowball Sampling
        srand(mpi.num_processes + mpi.rank);
        int start = rand() % neighbors.size();  // replace this with a proper int distribution
        std::cout << "rank: " << mpi.rank << " with start = " << start << std::endl;
        this->_my_vertices[start] = 1;
        for (int neighbor : neighbors[start]) {
            frontier.insert(neighbor);
        }
        int block = this->_block_assignment[start];
        this->_my_blocks[block] = true;
        int num_vertices = 1;
        while (num_vertices < target) {
            std::unordered_set<int> new_frontier;
            for (int vertex : frontier) {
                if (this->_my_vertices[vertex] == 1) continue;
                this->_my_vertices[vertex] = 1;
                for (int neighbor : neighbors[vertex]) {
                    new_frontier.insert(neighbor);
                }
                block = this->_block_assignment[vertex];
                this->_my_blocks[block] = true;
                num_vertices++;
                if (num_vertices == target) break;
            }
            if (num_vertices < target && frontier.size() == 0) {  // restart with a new vertex that isn't already selected
                std::unordered_set<int> candidates;
                for (int i = 0; i < (int) neighbors.size(); ++i) {
                    if (this->_my_vertices[i] == 0) candidates.insert(i);
                }
                int index = rand() % candidates.size();
                auto it = candidates.begin();
                std::advance(it, index);
                start = *it;
                this->_my_vertices[start] = 1;
                for (int neighbor : neighbors[start]) {
                    new_frontier.insert(neighbor);
                }
                block = this->_block_assignment[start];
                this->_my_blocks[block] = true;
                num_vertices++;
            }
            frontier = std::unordered_set<int>(new_frontier);
        }
        // Some vertices may be unassigned across all ranks. Find out what they are, and assign 1/num_processes of them
        // to this process.
        std::vector<int> global_selected(neighbors.size(), 0);
        MPI_Allreduce(this->_my_vertices.data(), global_selected.data(), neighbors.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // if (mpi.rank == 0) {
            // std::cout << "my selected: " << std::boolalpha;
            // utils::print<int>(this->_my_vertices);
            // std::cout << "globally selected: ";
            // utils::print<int>(global_selected);
        // }
        std::vector<int> vertices_left;
        for (int vertex = 0; vertex < (int) global_selected.size(); ++vertex) {
            if (global_selected[vertex] == 0) {
                vertices_left.push_back(vertex);
            }
        }
        // assign remaining vertices in round-robin fashion
        for (size_t i = mpi.rank; i < vertices_left.size(); i += mpi.num_processes) {
            int vertex = vertices_left[i];
            this->_my_vertices[vertex] = 1;
            block = this->_block_assignment[vertex];
            this->_my_blocks[block] = true;
        }
    }
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    this->build_two_hop_blockmodel(neighbors);
}

void merge(int vertex1, int vertex2, std::vector<int> &supernode_mapping, MapVector<bool> &available,
           std::unordered_map<int, std::vector<int>> &supernodes, MapVector<MapVector<bool>> &out_neighbors,
           MapVector<MapVector<bool>> &in_neighbors) {
    assert(available[vertex1] == true);
    assert(available[vertex2] == true);
    assert(supernode_mapping[vertex1] == vertex1);
    assert(supernode_mapping[vertex2] == vertex2);
//    if (mpi.rank == 0)
//        std::cout << "merging " << vertex2 << " (" << supernodes[vertex2].size() << ") into " << vertex1 << " (" << supernodes[vertex1].size() << ")" << std::endl;
    auto root = [&supernode_mapping](int vertex) -> int {
        int temp = supernode_mapping[vertex];
        while (supernode_mapping[temp] != temp) {
            temp = supernode_mapping[temp];
        }
        return temp;
    };
    // update mapping
    supernode_mapping[vertex2] = root(vertex1);
    // update supernodes
    std::vector<int> supernode2 = supernodes[vertex2];  // keeping this as a reference messed things up real bad
    for (int vertex : supernode2) {
        supernodes[vertex1].push_back(vertex);  // so did trying to use a reference to supernodes[vertex1]. go figure
    }
    supernodes.erase(vertex2);
    // update available
    available.erase(vertex1);
    available.erase(vertex2);
    // update neighbors
    for (const auto &elem : out_neighbors[vertex2]) {
        out_neighbors[vertex1][elem.first] = true;
    }
    for (const auto &elem : in_neighbors[vertex2]) {
        in_neighbors[vertex1][elem.first] = true;
    }
    for (const std::pair<int, bool> &element : in_neighbors[vertex2]) {
        out_neighbors[element.first].erase(vertex2);
        out_neighbors[element.first][vertex1] = true;
    }
    for (const std::pair<int, bool> &element : out_neighbors[vertex2]) {
        in_neighbors[element.first].erase(vertex2);
        in_neighbors[element.first][vertex1] = true;
    }
    out_neighbors.erase(vertex2);
    in_neighbors.erase(vertex2);
}

void TwoHopBlockmodel::distribute_2hop_super_snowball(const NeighborList &out_neighbors,
                                                      const NeighborList &in_neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    if (this->_my_vertices.size() == out_neighbors.size()) {  // if already done sampling, no need to do it again
        std::cout << "already done sampling, now just re-assigning blocks based on sampled vertices" << std::endl;
        for (size_t vertex = 0; vertex < out_neighbors.size(); ++vertex) {
            if (this->_my_vertices[vertex] == 0) continue;
            int block = this->_block_assignment[vertex];
            this->_my_blocks[block] = true;
        }
        this->build_two_hop_blockmodel(out_neighbors);
        return;
    }
    this->_my_vertices = utils::constant<int>(out_neighbors.size(), 0);
    MapVector<MapVector<bool>> out_neighbors_copy;
    MapVector<MapVector<bool>> in_neighbors_copy;
    MapVector<bool> available;
    std::vector<int> supernode_mapping = utils::range<int>(0, out_neighbors.size());
    std::unordered_map<int, std::vector<int>> supernodes;
    for (int vertex = 0; vertex < out_neighbors.size(); ++vertex) {
        available[vertex] = true;
        supernodes[vertex] = { vertex };
        out_neighbors_copy[vertex] = MapVector<bool>();
        in_neighbors_copy[vertex] = MapVector<bool>();
        for (int neighbor : out_neighbors[vertex]) {
            out_neighbors_copy[vertex][neighbor] = true;
        }
        for (int neighbor : in_neighbors[vertex]) {
            in_neighbors_copy[vertex][neighbor] = true;
        }
    }
    auto supernode_size = [&supernodes]() {
        int total = 0;
        for (const auto &supernode : supernodes) {
            total += supernode.second.size();
        }
        std::cout << "Total supernode size = " << total << std::endl;
    };
    while (supernodes.size() > 2 * mpi.num_processes) {
        if (mpi.rank == 0) {
            std::cout << "another iteration, supernodes with size = " << supernodes.size() << std::endl;
            supernode_size();
        }
        std::vector<int> supernode_indices;
        for (const std::pair<const int, std::vector<int>> &element : supernodes) {
            supernode_indices.push_back(element.first);
        }
        for (int vertex : supernode_indices) {
            if (!available[vertex]) continue;
            int found = false;
            const MapVector<bool> &neighbors = out_neighbors_copy[vertex];
            for (const std::pair<int, bool> &element : neighbors) {
                int neighbor = element.first;
                if (supernode_mapping[neighbor] != neighbor) continue;
                if (!available[neighbor]) continue;
                merge(vertex, neighbor, supernode_mapping, available, supernodes, out_neighbors_copy, in_neighbors_copy);
                found = true;
                break;
            }
            if (found == false && !available.empty()) {  // no available neighbors, merge with random (first available) vertex
                int random_vertex = -1;
                for (const std::pair<int, bool> &elem : available) {
                    if (!elem.second) continue;
                    random_vertex = elem.first;
                    break;
                }
                if (random_vertex < 0) continue;
                merge(vertex, random_vertex, supernode_mapping, available, supernodes, out_neighbors_copy, in_neighbors_copy);
            }
        }
        available.clear();
        for (const std::pair<const int, std::vector<int>> element : supernodes) {
            available[element.first] = true;
        }
//        std::cout << "Supernodes after iteration ==============" << std::endl;
//        for (const auto &supernode : supernodes) {
//            std::cout << supernode.first << " ";
//            utils::print(supernode.second);
//        }
    }
    if (mpi.rank == 0) {
        std::cout << "Supernodes at end ==============" << std::endl;
        for (const auto &supernode : supernodes) {
            std::cout << supernode.first << " | " << supernode.second.size() << std::endl;
//            std::cout << supernode.first << " ";
//            utils::print(supernode.second);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Reached the end yo!" << std::endl;
    exit(-600000);
    int index = 0;
    for (const std::pair<int, std::vector<int>> &supernode : supernodes) {
        if (index % mpi.rank != 0) continue;
        for (int vertex : supernode.second) {
            this->_my_vertices[vertex] = 1;
            this->_my_blocks[vertex] = true;
        }
    }
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    this->build_two_hop_blockmodel(out_neighbors);
}

void TwoHopBlockmodel::initialize_edge_counts(const NeighborList &neighbors) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    if (args.transpose) {
        this->_blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
    } else {
        this->_blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
    }
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    this->_block_degrees_in = utils::constant<int>(this->num_blocks, 0);
    this->_block_degrees_out = utils::constant<int>(this->num_blocks, 0);

    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.empty()) {
            continue;
        }
        int block = this->_block_assignment[vertex];
        if (!this->_in_two_hop_radius[block]) {
            continue;
        }
        for (size_t i = 0; i < vertex_neighbors.size(); ++i) {
            // Get count
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_block_assignment[neighbor];
            if (!this->_in_two_hop_radius[neighbor_block]) {
                continue;
            }
            // TODO: change this once code is updated to support weighted graphs
            int weight = 1;
            // Update blockmodel
            this->_blockmatrix->add(block, neighbor_block, weight);
            // Update degrees
            this->_block_degrees_out[block] += weight;
            this->_block_degrees_in[neighbor_block] += weight;
        }
    }
    // Count block degrees
    if (args.undirected) {
        this->_block_degrees = std::vector<int>(this->_block_degrees_out);
    } else {
        this->_block_degrees = this->_block_degrees_out + this->_block_degrees_in;
    }
}

double TwoHopBlockmodel::log_posterior_probability() const {
    std::vector<int> my_blocks;
    if (args.distribute == "2hop-snowball") {
        my_blocks = utils::constant<int>(this->num_blocks, -1);
        for (int block = 0; block < this->num_blocks; ++block) {
            if (this->_my_blocks[block])
                my_blocks[block] = mpi.rank;
        }
        MPI_Allreduce(MPI_IN_PLACE, my_blocks.data(), this->num_blocks, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // utils::print<int>(my_blocks);
    }
    Indices nonzero_indices = this->_blockmatrix->nonzero();
    std::vector<double> all_values = utils::to_double<int>(this->_blockmatrix->values());
    std::vector<double> degrees_in;
    std::vector<double> degrees_out;
    std::vector<double> values;
    for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
        int row = nonzero_indices.rows[i];
        if (args.distribute == "2hop-snowball") {
            if (my_blocks[row] != mpi.rank) continue;
        } else {
            if (this->_my_blocks[row] == false) continue;
        }
        // if (row % mpi.num_processes != mpi.rank) continue;
        values.push_back(all_values[i]);
        degrees_in.push_back(this->_block_degrees_in[nonzero_indices.cols[i]]);
        degrees_out.push_back(this->_block_degrees_out[nonzero_indices.rows[i]]);
    }
    std::vector<double> temp = values * utils::nat_log<double>(
        values / (degrees_out * degrees_in));
    double partial_sum = utils::sum<double>(temp);
    // MPI COMMUNICATION START
    double final_sum = 0.0;
    MPI_Allreduce(&partial_sum, &final_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI COMMUNICATION END
    // Alternative Plan for sampled 2-hop blockmodel:
    // 1. Break all_values, degrees_in, degrees_out into row-like statuses
    // 2. Compute temp across the rows that you own
    // 3. Perform an AllReduce MAX to find missing values
    // 4. Sum across the rows to find the final_sum
    return final_sum;
}

bool TwoHopBlockmodel::owns_block(int block) const {
    return this->_my_blocks[block];
}

bool TwoHopBlockmodel::owns_vertex(int vertex) const {
    if (args.distribute == "2hop-snowball") {
        return this->_my_vertices[vertex];
    }
    int block = this->_block_assignment[vertex];
    return this->owns_block(block);
}

std::vector<std::pair<int,int>> TwoHopBlockmodel::sorted_block_sizes() const {
    std::vector<std::pair<int,int>> block_sizes;
    for (int i = 0; i < this->num_blocks; ++i) {
        block_sizes.push_back(std::make_pair(i, 0));
    }
    for (const int &block : this->_block_assignment) {
        block_sizes[block].second++;
    }
    std::sort(block_sizes.begin(), block_sizes.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.second > b.second; });
    return block_sizes;
}

bool TwoHopBlockmodel::stores(int block) const {
    return this->_in_two_hop_radius[block];
}
