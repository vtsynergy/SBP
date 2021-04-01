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
                                       const std::vector<int> &best_merge_for_each_block, const Graph &graph) {
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
            this->merge_blocks(merge_from, merge_to, graph);
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

Blockmodel Blockmodel::clone_with_true_block_membership(const Graph &graph, std::vector<int> &true_block_membership) {
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
    return Blockmodel(num_blocks, graph, this->block_reduction_rate, true_block_membership);
}

Blockmodel Blockmodel::copy() {
    Blockmodel blockmodel_copy = Blockmodel(this->num_blocks, this->block_reduction_rate);
    blockmodel_copy.block_assignment = std::vector<int>(this->block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy.blockmodel = this->blockmodel.copy();
    blockmodel_copy.block_degrees = std::vector<int>(this->block_degrees);
    blockmodel_copy.block_degrees_out = std::vector<int>(this->block_degrees_out);
    blockmodel_copy.block_degrees_in = std::vector<int>(this->block_degrees_in);
    blockmodel_copy._block_degree_histograms = std::vector<DegreeHistogram>(this->_block_degree_histograms);
    blockmodel_copy._block_sizes = std::vector<int>(this->_block_sizes);
    blockmodel_copy.sampler = this->sampler.copy();
    // Create a Sampler as well?
    blockmodel_copy.num_blocks_to_merge = 0;
    return blockmodel_copy;
}

Blockmodel Blockmodel::from_sample(int num_blocks, const Graph &graph, std::vector<int> &sample_block_membership,
                                   std::map<int, int> &mapping, float block_reduction_rate) {
    // Fill in initial block assignment
    std::vector<int> block_assignment = utils::constant<int>(graph.num_vertices, -1);
    for (const auto &item : mapping) {
        block_assignment[item.first] = sample_block_membership[item.second];
    }
    // Every unassigned block gets assigned to the next block number
    int next_block = num_blocks;
    for (uint vertex = 0; vertex < graph.num_vertices; ++vertex) {
        if (block_assignment[vertex] >= 0) {
            continue;
        }
        block_assignment[vertex] = next_block;
        next_block++;
    }
    // Every previously unassigned block gets assigned to the block it's most connected to
    for (uint vertex = 0; vertex < graph.num_vertices; ++vertex) {
        if (block_assignment[vertex] < num_blocks) {
            continue;
        }
        std::vector<int> block_counts = utils::constant<int>(num_blocks, 0);
        // TODO: this can only handle unweighted graphs
        std::vector<int> vertex_neighbors = graph.out_neighbors[vertex];
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
    return Blockmodel(num_blocks, graph, block_reduction_rate, block_assignment);
}

void Blockmodel::initialize_edge_counts(const Graph &graph) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    this->blockmodel = DictTransposeMatrix(this->num_blocks, this->num_blocks);
    this->sampler = Sampler(this->num_blocks);
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    this->_block_degree_histograms = std::vector<DegreeHistogram>(this->num_blocks, DegreeHistogram());
    this->_block_sizes = std::vector<int>(this->num_blocks, 0);
    this->block_degrees_in = utils::constant<int>(this->num_blocks, 0);
    this->block_degrees_out = utils::constant<int>(this->num_blocks, 0);
    // Initialize the blockmodel
    // TODO: find a way to parallelize the matrix filling step
    assert(graph.out_neighbors.size() == graph.num_vertices);
    for (uint vertex = 0; vertex < graph.out_neighbors.size(); ++vertex) {
        std::vector<int> vertex_out_neighbors = graph.out_neighbors[vertex];
        int block = this->block_assignment[vertex];
        auto degree_pair = std::make_pair(graph.in_neighbors[vertex].size(), graph.out_neighbors[vertex].size());
        this->_block_degree_histograms[block][degree_pair]++;
        this->_block_sizes[block]++;
        if (vertex_out_neighbors.size() == 0) {
            continue;
        }
        // int block = this->block_assignment[vertex];
        for (int i = 0; i < vertex_out_neighbors.size(); ++i) {
            // Get count
            int neighbor = vertex_out_neighbors[i];
            int neighbor_block = this->block_assignment[neighbor];
            // TODO: change this once code is updated to support weighted graphs
            int weight = 1;
            // int weight = vertex_out_neighbors[i];
            // Update blockmodel
            this->blockmodel.add(block, neighbor_block, weight);
            // Update degrees
            this->block_degrees_out[block] += weight;
            this->block_degrees_in[neighbor_block] += weight;
            this->sampler.insert(block, neighbor_block);
        }
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

void Blockmodel::merge_blocks(int from_block, int to_block, const Graph &graph) {
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        if (this->block_assignment[vertex] == from_block) {
            this->block_assignment[vertex] = to_block;
            auto degree_pair = std::make_pair(graph.in_neighbors[vertex].size(), graph.out_neighbors[vertex].size());
            this->_block_degree_histograms[from_block][degree_pair]--;
            this->_block_degree_histograms[to_block][degree_pair]++;
        }
    }
    this->_block_sizes[to_block] += this->_block_sizes[from_block];
    this->_block_sizes[from_block] = 0;
    MapVector<int> neighbors = this->sampler.neighbors(from_block);
    for (std::pair<int, int> neighbor : neighbors) {
        if (neighbor.first == from_block) {
            this->sampler.insert(to_block, to_block, neighbor.second);
            this->sampler.remove(from_block, from_block, neighbor.second);
            continue;
        }
        this->sampler.insert(neighbor.first, to_block, neighbor.second);
        this->sampler.remove(neighbor.first, from_block, neighbor.second);
    }
}

void Blockmodel::move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                             std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                             std::vector<int> &new_block_degrees, const Graph &graph) {
    this->block_assignment[vertex] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->block_degrees_out = new_block_degrees_out;
    this->block_degrees_in = new_block_degrees_in;
    this->block_degrees = new_block_degrees;
    auto degree_pair = std::make_pair(graph.in_neighbors[vertex].size(), graph.out_neighbors[vertex].size());
    this->_block_degree_histograms[current_block][degree_pair]--;
    this->_block_degree_histograms[new_block][degree_pair]++;
    this->_block_sizes[current_block]--;
    this->_block_sizes[new_block]++;
}

void Blockmodel::move_vertex_delta(int vertex, int current_block, int new_block, EntryMap &deltas,
                                   std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                                   std::vector<int> &new_block_degrees, const Graph &graph) {
    this->block_assignment[vertex] = new_block;
    for (const std::pair<std::pair<int, int>, int> &delta: deltas) {
        int row = delta.first.first;
        int col = delta.first.second;
        this->blockmodel.add(row, col, delta.second);
        if (this->blockmodel.get(row, col) < 0) {
            std::cout << "ERROR!! A! vertex: " << vertex << " (" << current_block << " --> " << new_block << ") [";
            std::cout << row << "," << col << "] delta = " << delta.second << std::endl;;
            exit(-10);
        }
    }
    this->block_degrees_out = new_block_degrees_out;
    this->block_degrees_in = new_block_degrees_in;
    this->block_degrees = new_block_degrees;
    auto degree_pair = std::make_pair(graph.in_neighbors[vertex].size(), graph.out_neighbors[vertex].size());
    this->_block_degree_histograms[current_block][degree_pair]--;
    this->_block_degree_histograms[new_block][degree_pair]++;
    this->_block_sizes[current_block]--;
    this->_block_sizes[new_block]++;
    for (int out_neighbor : graph.out_neighbors[vertex]) {
        if (out_neighbor == vertex) {
            this->sampler.insert(new_block, new_block, 1);
            this->sampler.remove(current_block, current_block, 1);
            continue;
        }
        int block = this->block_assignment[out_neighbor];
        this->sampler.insert(new_block, block, 1);
        this->sampler.remove(current_block, block, 1);
    }
    for (int in_neighbor : graph.in_neighbors[vertex]) {
        if (in_neighbor == vertex) continue;  // already handled above
        int block = this->block_assignment[in_neighbor];
        // if (block == current_block) {
        //     this->sampler.insert(new_block, new_block, 1);
        //     this->sampler.remove(current_block, current_block, 1);
        //     continue;
        // }
        this->sampler.insert(new_block, block, 1);
        this->sampler.remove(current_block, block, 1);
    }
}

void Blockmodel::move_vertex_delta(int vertex, int current_block, int new_block, SparseEdgeCountUpdates &delta,
                                   std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                                   std::vector<int> &new_block_degrees, const Graph &graph) {
    this->block_assignment[vertex] = new_block;
    for (const std::pair<int, int> &delta : delta.block_row) {
        this->blockmodel.add(current_block, delta.first, delta.second);
        if (this->blockmodel.get(current_block, delta.first) < 0) {
            std::cout << "ERROR!! A! vertex: " << vertex << " current_block: " << current_block;
            std::cout << " new_block: " << new_block << " delta.first: " << delta.first << " delta.second: " << delta.second;
            exit(-10);
        }
    }
    for (const std::pair<int, int> &delta : delta.block_col) {
        this->blockmodel.add(delta.first, current_block, delta.second);
        if (this->blockmodel.get(delta.first, current_block) < 0) {
            std::cout << "ERROR!! B! vertex: " << vertex << " current_block: " << current_block;
            std::cout << " new_block: " << new_block << " delta.first: " << delta.first << " delta.second: " << delta.second;
            exit(-10);
        }
    }
    for (const std::pair<int, int> &delta : delta.proposal_row) {
        this->blockmodel.add(new_block, delta.first, delta.second);
        if (this->blockmodel.get(new_block, delta.first) < 0) {
            std::cout << "ERROR!! C! vertex: " << vertex << " current_block: " << current_block;
            std::cout << " new_block: " << new_block << " delta.first: " << delta.first << " delta.second: " << delta.second;
            exit(-10);
        }
    }
    for (const std::pair<int, int> &delta : delta.proposal_col) {
        this->blockmodel.add(delta.first, new_block, delta.second);
        if (this->blockmodel.get(delta.first, new_block) < 0) {
            std::cout << "ERROR!! D! vertex: " << vertex << " current_block: " << current_block;
            std::cout << " new_block: " << new_block << " delta.first: " << delta.first << " delta.second: " << delta.second;
            exit(-10);
        }
    }
    this->block_degrees_out = new_block_degrees_out;
    this->block_degrees_in = new_block_degrees_in;
    this->block_degrees = new_block_degrees;
    auto degree_pair = std::make_pair(graph.in_neighbors[vertex].size(), graph.out_neighbors[vertex].size());
    this->_block_degree_histograms[current_block][degree_pair]--;
    this->_block_degree_histograms[new_block][degree_pair]++;
    this->_block_sizes[current_block]--;
    this->_block_sizes[new_block]++;
}

// TODO: computing the entries first would be inefficient
EntryMap Blockmodel::deltas(int current_block, int proposed_block, const EntryMap &entries) {
    EntryMap result;
    for (const std::pair<std::pair<int, int>, int> &entry : entries) {
        int row = entry.first.first;
        int col = entry.first.second;
        int weight = entry.second;
        result[entry.first] -= weight;
        if (row == current_block && col == current_block)  // entry = M[current_block, proposed_block]
            result[std::make_pair(proposed_block, proposed_block)] += weight;
        else if (row == proposed_block || col == proposed_block)  // entry = M[proposed_block, X] | M[X, proposed_block]
            result[std::make_pair(proposed_block, proposed_block)] += weight;
        else if (row == current_block)  // entry = M[current_block, X]
            result[std::make_pair(proposed_block, col)] += weight;
        else if (col == current_block)  // entry = M[X, current_block]
            result[std::make_pair(row, proposed_block)] += weight;
    }
    return result;
}

EntryMap Blockmodel::entries1(int block, int exclude) {
    EntryMap result;
    // const MapVector<int> &rowA = this->blockmodel.getrow_sparse(blockA);
    for (const std::pair<int,int> &entry : this->blockmodel.getrow_sparse(block)) {
        if (entry.first == exclude) continue;
        result[std::make_pair(block, entry.first)] = entry.second;
    }
    for (const std::pair<int,int> &entry : this->blockmodel.getcol_sparse(block)) {
        if (entry.first == block || entry.first == exclude) continue;
        result[std::make_pair(entry.first, block)] = entry.second;
    }
    return result;
}

EntryMap Blockmodel::entries2(int blockA, int blockB) {
    EntryMap result;
    // const MapVector<int> &rowA = this->blockmodel.getrow_sparse(blockA);
    for (const std::pair<int,int> &entry : this->blockmodel.getrow_sparse(blockA))
        result[std::make_pair(blockA, entry.first)] = entry.second;
    for (const std::pair<int,int> &entry : this->blockmodel.getcol_sparse(blockA)) {
        if (entry.first == blockA) continue;
        result[std::make_pair(entry.first, blockA)] = entry.second;
    }
    for (const std::pair<int,int> &entry : this->blockmodel.getrow_sparse(blockB)) {
        if (entry.first == blockA) continue;
        result[std::make_pair(blockB, entry.first)] = entry.second;
    }
    for (const std::pair<int,int> &entry : this->blockmodel.getcol_sparse(blockB)) {
        if (entry.first == blockA || entry.first == blockB) continue;
        result[std::make_pair(entry.first, blockB)] = entry.second;
    }
    return result;
}

void Blockmodel::print() {
    std::cout << "blockmodel: " << std::endl;
    for (int row = 0; row < this->num_blocks; ++row) {
        if (row < 10) {
            std::cout << " " << row << " | ";
        } else {
            std::cout << row << " | ";
        }
        for (int val : this->blockmodel.getrow(row)) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int Blockmodel::sample(int block, std::mt19937_64 &generator) {
    return this->sampler.sample(block, generator);
}

void Blockmodel::set_block_membership(int vertex, int block, const Graph &graph) {
    int current_block = this->block_assignment[vertex];
    this->block_assignment[vertex] = block;
    auto degree_pair = std::make_pair(graph.in_neighbors[vertex].size(), graph.out_neighbors[vertex].size());
    if (current_block >= 0) {
        this->_block_degree_histograms[current_block][degree_pair]--;
        this->_block_sizes[current_block]--;
    }
    this->_block_degree_histograms[block][degree_pair]++;
    this->_block_sizes[block]++;
}

void Blockmodel::update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates) {
    this->blockmodel.update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                        updates.block_col, updates.proposal_col);
}

bool Blockmodel::operator==(const Blockmodel &other) {
    if (this->num_blocks != other.num_blocks) return false;
    for (int block = 0; block < this->num_blocks; ++block) {
        if (this->block_assignment[block] != other.block_assignment[block]) return false;
        if (this->block_degrees[block] != other.block_degrees[block]) return false;
        if (this->block_degrees_in[block] != other.block_degrees_in[block]) return false;
        if (this->block_degrees_out[block] != other.block_degrees_out[block]) return false;
        if (this->_block_sizes[block] != other._block_sizes[block]) return false;
        // degree histogram
        for (const std::pair<std::pair<int, int>, int> &degree : this->_block_degree_histograms[block]) {
            auto count = other._block_degree_histograms[block].find(degree.first);
            if (count == other._block_degree_histograms[block].end() && degree.second != 0) {
                assert(false);
                return false;
            }
            if (degree.second != count->second) return false;
        }
        for (const std::pair<std::pair<int, int>, int> &degree : other._block_degree_histograms[block]) {
            if (degree.second != this->_block_degree_histograms[block][degree.first]) {
                assert(false);
                return false;
            }
        }
        // sampler
        for (const std::pair<int, int> &neighbor : this->sampler.neighbors(block)) {
            auto count = other.sampler.neighbors(block).find(neighbor.first);
            if (count == other.sampler.neighbors(block).end()) {
                std::cout << this << " sampler doesn't have " << neighbor.first << " : " << neighbor.second << std::endl;
                assert(false);
                return false;
            }
            if (neighbor.second != count->second) return false;
        }
        for (const std::pair<int, int> &neighbor : other.sampler.neighbors(block)) {
            auto count = this->sampler.neighbors(block).find(neighbor.first);
            if (count == this->sampler.neighbors(block).end()) {
                std::cout << this << " sampler doesn't have " << neighbor.first << " : " << neighbor.second << std::endl;
                assert(false);
                return false;
            }
            if (neighbor.second != count->second) return false;
        }
        // if (this->sampler.neighbors(block) != other.sampler.neighbors(block)) {
        //     assert(false);
        //     return false;
        // }
    }
    for (int row = 0; row < this->num_blocks; ++row) {
        for (int col = 0; col < this->num_blocks; ++col) {
            if (this->blockmodel.get(row, col) != other.blockmodel.get(row, col)) return false;
        }
    }
    return true;
}

void Blockmodel::assert_stats() {
    std::cout << "asserting stats!" << std::endl;
    std::cout << "checking _block_sizes" << std::endl;
    int total = 0;
    for (int s : this->_block_sizes) {
        total += s;
    }
    assert (total == this->block_assignment.size());
    std::cout << "checking _block_degree_histograms" << std::endl;
    total = 0;
    for (int b = 0; b < this->num_blocks; ++b) {
        for (auto pair : this->_block_degree_histograms[b]) {
            total += pair.second;
        }
    }
    assert (total == this->block_assignment.size());
    std::cout << "ALL GOOD!!!! WOOOO!!!!" << std::endl;
}

Sampler Sampler::copy() {
    // Sampler sampler_copy = Sampler(0);
    // for (int i = 0; i < this->_num_blocks; ++i) {
    //     const MapVector<int> &neighborhood = this->_neighbors[i];
    //     MapVector<int> new_neighborhood;
    //     for (const std::pair<int, int> &entry : neighborhood) {
    //         new_neighborhood[entry.first] = entry.second;
    //     }
    //     sampler_copy._neighbors.push_back(new_neighborhood);
    // }
    // sampler_copy._num_blocks = this->_num_blocks;
    Sampler sampler_copy = Sampler(this->_num_blocks);
    for (int i = 0; i < this->_num_blocks; ++i) {
        const MapVector<int> &neighborhood = this->_neighbors[i];
        sampler_copy._neighbors[i] = MapVector<int>(neighborhood);
    }
    return sampler_copy;
}

void Sampler::insert(int from, int to, int count) {
    // if (from == to) return;
    this->_neighbors[from][to] += count;
    if (from == to) return;
    this->_neighbors[to][from] += count;
    // this->_neighbors[from].insert(to);
    // this->_neighbors[to].insert(from);
}

void Sampler::remove(int from, int to, int count) {
    // std::cout << "from: " << from << "to: " << to << " count: " << count << " actual: " << this->_neighbors[from][to] << std::endl;
    if (this->_neighbors[from][to] == count)
        this->_neighbors[from].erase(to);
    else
        this->_neighbors[from][to] -= count;
    if (from == to) return;
    if (this->_neighbors[to][from] == count)
        this->_neighbors[to].erase(from);
    else
        this->_neighbors[to][from] -= count;
    // this->_neighbors[from].erase(to);
    // this->_neighbors[to].erase(from);
}

int Sampler::sample(int block, std::mt19937_64 &generator) {
    const MapVector<int> &neighborhood = this->_neighbors[block];
    // const std::set<int> &neighborhood = this->_neighbors[block];
    // if (neighborhood.empty()) {  // sample a random block
    //     std::uniform_int_distribution<int> distribution(0, this->num_blocks - 2);
    //     int sampled = distribution(generator);
    //     if (sampled >= block) {
    //         sampled++;
    //     }
    //     return sampled;
    // }
    if (neighborhood.size() == 0)
        return NULL_BLOCK;
    std::uniform_int_distribution<int> distribution(0, neighborhood.size() - 1);
    int index = distribution(generator);
    // std::set doesn't have access by index - use iterator instead.
    auto it = neighborhood.begin();
    // auto it = this->_neighbors[block].begin();
    // MapVector<int>::const_iterator it = neighborhood.begin();
    // std::set<int>::iterator it = neighborhood.begin();
    // std::cout << "sampling for block: " << block << " index: " << index << " iterator: it[" << it->first << " " << it->second << "] neighborhood: (" << neighborhood.size() << ") ";
    std::advance(it, index);
    // utils::print<int>(neighborhood);
    int sampled = it->first;
    return sampled;
}
