#include "distributed/two_hop_blockmodel.hpp"

#include <unordered_set>

void TwoHopBlockmodel::build_two_hop_blockmodel(const NeighborList &neighbors) {
    if (args.distribute == "none" || args.distribute == "none-edge-balanced" ||
        args.distribute == "none-agg-block-degree-balanced") {
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
    else if (args.distribute == "none-edge-balanced")
        distribute_none_edge_balanced(graph);
    else if (args.distribute == "none-agg-block-degree-balanced")
        distribute_none_agg_block_degree_balanced(graph);
    else
        distribute_none();
    if (args.distribute != "none" && args.distribute != "none-edge-balanced" &&
        args.distribute != "none-agg-block-degree-balanced") {
        std::cout << "WARNING: data distribution is NOT fully supported yet. "
                  << "We STRONGLY recommend running this software with --distribute none instead" << std::endl;
    }
}

void TwoHopBlockmodel::distribute_none() {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    for (int i = mpi.rank; i < this->num_blocks; i += mpi.num_processes)
        this->_my_blocks[i] = true;
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_none_edge_balanced(const Graph &graph) {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    this->_my_vertices = utils::constant<int>(graph.num_vertices(), 0);
    std::vector<int> vertex_degrees = graph.degrees();
    std::vector<int> sorted_indices = utils::sort_indices<int>(vertex_degrees);
    for (int i = mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
        int vertex = sorted_indices[i];
        this->_my_vertices[vertex] = 1;
    }
    for (int i = 2 * mpi.num_processes - 1 - mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
        int vertex = sorted_indices[i];
        this->_my_vertices[vertex] = 1;
    }
    std::vector<std::pair<int,int>> block_sizes = this->sorted_block_sizes();
    for (int i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    for (int i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_none_agg_block_degree_balanced(const Graph &graph) {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    this->_my_vertices = utils::constant<int>(graph.num_vertices(), 0);
    std::vector<int> block_degrees = utils::constant<int>(graph.num_vertices(), 0);
    for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        int block = this->_block_assignment[vertex];
        block_degrees[vertex] = this->_block_degrees[block];
    }
    std::vector<int> sorted_indices = utils::sort_indices<int>(block_degrees);
    for (int i = mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
        int vertex = sorted_indices[i];
        this->_my_vertices[vertex] = 1;
    }
    for (int i = 2 * mpi.num_processes - 1 - mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
        int vertex = sorted_indices[i];
        this->_my_vertices[vertex] = 1;
    }
    std::vector<std::pair<int,int>> block_sizes = this->sorted_block_sizes();
    for (int i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    for (int i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
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

void TwoHopBlockmodel::initialize_edge_counts(const Graph &graph) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    std::shared_ptr<ISparseMatrix> blockmatrix;
    int num_buckets = graph.num_edges() / graph.num_vertices();
    if (args.transpose) {
        blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks, num_buckets);
    } else {
        blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
    }
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    std::vector<int> block_degrees_in = utils::constant<int>(this->num_blocks, 0);
    std::vector<int> block_degrees_out = utils::constant<int>(this->num_blocks, 0);
    std::vector<int> block_degrees = utils::constant<int>(this->num_blocks, 0);
    // Initialize the blockmodel in parallel
    #pragma omp parallel default(none) \
    shared(blockmatrix, block_degrees_in, block_degrees_out, block_degrees, graph, args)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int my_num_blocks = ceil(double(this->num_blocks) / double(num_threads));
        int start = my_num_blocks * tid;
        int end = start + my_num_blocks;
        for (uint vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            int block = this->_block_assignment[vertex];
            if (block < start || block >= end || !this->_in_two_hop_radius[block])  // only modify blocks this thread is responsible for
                continue;
            for (int neighbor : graph.out_neighbors(int(vertex))) {
                int neighbor_block = this->_block_assignment[neighbor];
                if (!this->_in_two_hop_radius[neighbor_block]) {
                    continue;
                }
                int weight = 1;
                blockmatrix->add(block, neighbor_block, weight);
                block_degrees_out[block] += weight;
                block_degrees[block] += weight;
            }
            for (int neighbor : graph.in_neighbors(int(vertex))) {
                int neighbor_block = this->_block_assignment[neighbor];
                if (!this->_in_two_hop_radius[neighbor_block]) {
                    continue;
                }
                int weight = 1;
                if (args.transpose) {
                    std::shared_ptr<DictTransposeMatrix> blockmatrix_dtm =
                            std::dynamic_pointer_cast<DictTransposeMatrix>(blockmatrix);
                    blockmatrix_dtm->add_transpose(neighbor_block, block, weight);
                }
                block_degrees_in[block] += weight;
                if (block != neighbor_block) {
                    block_degrees[block] += weight;
                }
            }
        }
    }  // OMP_PARALLEL
    this->_blockmatrix = std::move(blockmatrix);
    this->_block_degrees_out = std::move(block_degrees_out);
    this->_block_degrees_in = std::move(block_degrees_in);
    this->_block_degrees = std::move(block_degrees);
}

double TwoHopBlockmodel::log_posterior_probability() const {
    std::vector<int> my_blocks;
    if (args.distribute == "2hop-snowball" || args.distribute == "none-edge-balanced" ||
        args.distribute == "none-agg-block-degree-balanced") {
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
    if (args.distribute == "2hop-snowball" || args.distribute == "none-edge-balanced" ||
        args.distribute == "none-agg-block-degree-balanced") {
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

bool TwoHopBlockmodel::validate(const Graph &graph) const {
    std::cout << "Validating..." << std::endl;
    std::vector<int> assignment(this->_block_assignment);
    Blockmodel correct(this->num_blocks, graph, this->block_reduction_rate, assignment);
    for (int row = 0; row < this->num_blocks; ++row) {
        for (int col = 0; col < this->num_blocks; ++col) {
            if (!(this->in_two_hop_radius()[row] || this->in_two_hop_radius()[col])) continue;
//            int this_val = this->blockmatrix()->get(row, col);
            int correct_val = correct.blockmatrix()->get(row, col);
            if (!this->blockmatrix()->validate(row, col, correct_val)) {
                std::cout << "ERROR::matrix[" << row << "," << col << "] is " << this->blockmatrix()->get(row, col) <<
                          " but should be " << correct_val << std::endl;
                return false;
            }
//            if (this_val != correct_val) return false;
        }
    }
    for (int block = 0; block < this->num_blocks; ++block) {
        bool valid = true;
        if (this->_block_degrees[block] != correct.degrees(block)) {
            std::cout << "ERROR::block degrees of " << block << " is " << this->_block_degrees[block] <<
                      " when it should be " << correct.degrees(block) << std::endl;
            valid = false;
        }
        if (this->_block_degrees_out[block] != correct.degrees_out(block)) {
            std::cout << "ERROR::block out-degrees of " << block << " is " << this->_block_degrees_out[block] <<
                      " when it should be " << correct.degrees_out(block) << std::endl;
            valid = false;
        }
        if (this->_block_degrees_in[block] != correct.degrees_in(block)) {
            std::cout << "ERROR::block in-degrees of " << block << " is " << this->_block_degrees_in[block] <<
                      " when it should be " << correct.degrees_in(block) << std::endl;
            valid = false;
        }
        if (!valid) {
            std::cout << "ERROR::error state | d_out: " << this->_block_degrees_out[block] << " d_in: " <<
                      this->_block_degrees_in[block] << " d: " << this->_block_degrees[block] <<
                      " self_edges: " << this->blockmatrix()->get(block, block) << std::endl;
            std::cout << "ERROR::correct state | d_out: " << correct.degrees_out(block) << " d_in: " <<
                      correct.degrees_in(block) << " d: " << correct.degrees(block) <<
                      " self_edges: " << correct.blockmatrix()->get(block, block) << std::endl;
            std::cout << "ERROR::Checking matrix for errors..." << std::endl;
            for (int row = 0; row < this->num_blocks; ++row) {
                for (int col = 0; col < this->num_blocks; ++col) {
                    //            int this_val = this->blockmatrix()->get(row, col);
                    int correct_val = correct.blockmatrix()->get(row, col);
                    if (!this->blockmatrix()->validate(row, col, correct_val)) {
                        std::cout << "matrix[" << row << "," << col << "] is " << this->blockmatrix()->get(row, col) <<
                                  " but should be " << correct_val << std::endl;
                        return false;
                    }
                    //            if (this_val != correct_val) return false;
                }
            }
            std::cout << "ERROR::Block degrees not valid, but no errors were found in matrix" << std::endl;
            return false;
        }
    }
    return true;
}
