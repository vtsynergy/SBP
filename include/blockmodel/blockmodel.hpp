/***
 * Stores the current graph blockmodeling results.
 */
#ifndef SBP_PARTITION_HPP
#define SBP_PARTITION_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>

// #include <Eigen/Core>
// #include "sparse/boost_mapped_matrix.hpp"
#include "graph.hpp"
#include "sparse/dict_transpose_matrix.hpp"
#include "sparse/typedefs.hpp"
#include "../utils.hpp"

// typedef py::EigenDRef<Eigen::Matrix<int, Eigen::Dynamic, 2>> Matrix2Column;

typedef struct edge_count_updates_t {
    std::vector<int> block_row;
    std::vector<int> proposal_row;
    std::vector<int> block_col;
    std::vector<int> proposal_col;
} EdgeCountUpdates;

typedef struct sparse_edge_count_updates_t {
    MapVector<int> block_row;
    MapVector<int> proposal_row;
    MapVector<int> block_col;
    MapVector<int> proposal_col;
} SparseEdgeCountUpdates;

// If/when sampler gets it's own .hpp file, this should move there too.
static std::random_device seeder;
static std::default_random_engine generator(seeder());

class Sampler {
public:
    Sampler(int num_blocks) : num_blocks(num_blocks) {
        for (int i = 0; i < num_blocks; ++i) {
            this->neighbors.push_back(std::set<int>());
        }
    }
    // const std::set<int> &get_neighbors(int block) { return this->neighbors[block]; };
    void insert(int from, int to);
    int sample(int block);
private:
    std::vector<std::set<int>> neighbors;
    int num_blocks;
};

// See https://www.techiedelight.com/use-std-pair-key-std-unordered_map-cpp/
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};
typedef std::unordered_map<std::pair<int, int>, int, pair_hash> DegreeHistogram;

class Blockmodel {
  public:
    Blockmodel() : empty(true), sampler(Sampler(0)) {}
    Blockmodel(int num_blocks, float block_reduction_rate) : empty(false), sampler(Sampler(num_blocks)) {
        this->num_blocks = num_blocks;
        this->block_reduction_rate = block_reduction_rate;
        this->overall_entropy = std::numeric_limits<float>::max();
        this->blockmodel = DictTransposeMatrix(this->num_blocks, this->num_blocks);
        // Set the block assignment to be the range [0, this->num_blocks)
        this->block_assignment = utils::range<int>(0, this->num_blocks);
        this->_block_degree_histograms = std::vector<DegreeHistogram>(this->num_blocks, DegreeHistogram());
        this->_block_sizes = std::vector<int>(this->num_blocks, 0);
        // Number of blocks to merge
        this->num_blocks_to_merge = (int)(this->num_blocks * this->block_reduction_rate);
        // this->sampler = Sampler(num_blocks);
    }
    Blockmodel(int num_blocks, const Graph &graph, float block_reduction_rate)
        : Blockmodel(num_blocks, block_reduction_rate) {
        this->initialize_edge_counts(graph);
    }
    Blockmodel(int num_blocks, const Graph &graph, float block_reduction_rate, std::vector<int> &block_assignment)
        : Blockmodel(num_blocks, block_reduction_rate) {
        // Set the block assignment
        this->block_assignment = block_assignment;
        // Number of blocks to merge
        this->initialize_edge_counts(graph);
    }
    /// TODO
    std::vector<int> build_mapping(std::vector<int> &values);
    /// Performs the block merges with the highest change in entropy/MDL
    void carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                               const std::vector<int> &best_merge_for_each_block, const Graph &graph);
    /// TODO
    Blockmodel clone_with_true_block_membership(const Graph &graph, std::vector<int> &true_block_membership);
    /// Returns a copy of the current Blockmodel
    Blockmodel copy();
    /// TODO documentation
    // TODO: move block_reduction_rate to some constants file
    static Blockmodel from_sample(int num_blocks, const Graph &graph, std::vector<int> &sample_block_membership,
                                  std::map<int, int> &mapping, float block_reduction_rate);
    /// TODO
    void initialize_edge_counts(const Graph &graph);
    /// TODO
    double log_posterior_probability();
    /// TODO
    void merge_blocks(int from_block, int to_block, const Graph &graph);
    /// TODO
    void move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                     std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                     std::vector<int> &new_block_degrees, const Graph &graph);
    /// TODO
    void move_vertex_delta(int vertex, int current_block, int new_block, SparseEdgeCountUpdates &delta,
                           std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                           std::vector<int> &new_block_degrees, const Graph &graph);
    /// Samples a community for the current block's neighbors. If the current block has no neighbors, returns a random
    /// community.
    int sample(int block);
    /// Sets the block membership of `vertex` to `block`.
    void set_block_membership(int vertex, int block, const Graph &graph);
    /// Updates the blockmodel matrix by replacing the appropriate rows and columns with those in `updates`.
    void update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates);
    /// TODO: Get rid of getters and setters?
    DictTransposeMatrix &getBlockmodel() { return this->blockmodel; }
    void setBlockmodel(DictTransposeMatrix blockmodel) { this->blockmodel = blockmodel; }
    std::vector<int> &getBlock_assignment() { return this->block_assignment; }
    void setBlock_assignment(std::vector<int> block_assignment) { this->block_assignment = block_assignment; }
    std::vector<int> &getBlock_degrees() { return this->block_degrees; }
    void setBlock_degrees(std::vector<int> block_degrees) { this->block_degrees = block_degrees; }
    std::vector<int> &getBlock_degrees_in() { return this->block_degrees_in; }
    void setBlock_degrees_in(std::vector<int> block_degrees_in) { this->block_degrees_in = block_degrees_in; }
    std::vector<int> &getBlock_degrees_out() { return this->block_degrees_out; }
    void setBlock_degrees_out(std::vector<int> block_degrees_out) { this->block_degrees_out = block_degrees_out; }
    float &getBlock_reduction_rate() { return this->block_reduction_rate; }
    void setBlock_reduction_rate(float block_reduction_rate) { this->block_reduction_rate = block_reduction_rate; }
    int block_size(int block) { return this->_block_sizes[block]; }
    std::vector<int> &block_sizes() { return this->_block_sizes; }
    DegreeHistogram &degree_histogram(int block) { return this->_block_degree_histograms[block]; }
    float &getOverall_entropy() { return this->overall_entropy; }
    void setOverall_entropy(float overall_entropy) { this->overall_entropy = overall_entropy; }
    int &getNum_blocks_to_merge() { return this->num_blocks_to_merge; }
    void setNum_blocks_to_merge(int num_blocks_to_merge) { this->num_blocks_to_merge = num_blocks_to_merge; }
    int &getNum_blocks() { return this->num_blocks; }
    void setNum_blocks(int num_blocks) { this->num_blocks = num_blocks; }
    // Other
    bool empty;
    void assert_stats();

  private:
    // Structure
    int num_blocks;
    DictTransposeMatrix blockmodel;
    // Known info
    std::vector<int> block_assignment;
    std::vector<int> block_degrees;
    std::vector<int> block_degrees_in;
    std::vector<int> block_degrees_out;
    std::vector<DegreeHistogram> _block_degree_histograms;
    std::vector<int> _block_sizes;
    float block_reduction_rate;
    // Computed info
    float overall_entropy;
    int num_blocks_to_merge;
    Sampler sampler;
    /// Sorts the indices of an array in descending order according to the values of the array
    std::vector<int> sort_indices(const std::vector<double> &unsorted);
};

#endif // SBP_PARTITION_HPP
