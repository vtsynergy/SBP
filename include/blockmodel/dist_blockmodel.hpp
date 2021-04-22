/***
 * Stores the current distributed graph blockmodeling results.
 */
#ifndef SBP_DIST_BLOCKMODEL_HPP
#define SBP_DIST_BLOCKMODEL_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <memory>
#include <mpi.h>
#include <queue>

// #include <Eigen/Core>
// #include "sparse/boost_mapped_matrix.hpp"
#include "../args.hpp"
#include "blockmodel.hpp"
#include "../graph.hpp"
#include "mpi_data.hpp"
#include "sparse/dict_matrix.hpp"
#include "sparse/dict_transpose_matrix.hpp"
#include "sparse/dist_dict_matrix.hpp"
#include "sparse/typedefs.hpp"
#include "../utils.hpp"

class DistBlockmodel {
  public:
    DistBlockmodel() : empty(true) {}
    // DistBlockmodel(int num_blocks, int global_num_blocks) : empty(false) {
    //     this->_num_blocks = num_blocks;
    //     this->_global_num_blocks = global_num_blocks;
    //     // this->block_reduction_rate = block_reduction_rate;
    //     this->_overall_entropy = std::numeric_limits<float>::max();
    //     // this->blockmodel = std::make_unique<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
    //     this->_blockmatrix = new DistDictMatrix(this->_global_num_blocks, this->_global_num_blocks);
    //     // Set the block assignment to be the range [0, this->num_blocks)
    //     this->_assignment = utils::range<int>(0, this->_num_blocks);

    //     // Number of blocks to merge
    //     this->_num_blocks_to_merge = (int)(this->_num_blocks * BLOCK_REDUCTION_RATE);
    // }
    // DistBlockmodel(int num_blocks, int global_num_blocks, const NeighborList &out_neighbors)
    //      : DistBlockmodel(num_blocks, global_num_blocks) {
    //     this->initialize_edge_counts(out_neighbors);
    // }
    // DistBlockmodel(int num_blocks, int global_num_blocks, const NeighborList &out_neighbors,
    //     std::vector<int> &block_assignment) : DistBlockmodel(num_blocks, global_num_blocks) {
    //     // Set the block assignment
    //     this->_assignment = block_assignment;
    //     // Number of blocks to merge
    //     this->initialize_edge_counts(out_neighbors);
    // }
    // DistBlockmodel(const Graph &graph, const Args &args, const MPI_Data &mpi) {
    DistBlockmodel(const Graph &graph, const Args &args) {
        /// ================= TODO: move to partition() methid =====================
        // NeighborList in_neighbors(graph.num_vertices());
        NeighborList out_neighbors(graph.num_vertices());
        int num_vertices = 0, num_edges = 0;
        std::unordered_map<int, int> translator;
        for (int i = mpi.rank; i < graph.out_neighbors().size(); i += mpi.num_processes) {
            if (utils::insert(translator, i, num_vertices))
                num_vertices++;
            int from = i;
            // int from = translator[i];  // TODO: can avoid additional lookups by returning the inserted element in insert
            for (int neighbor : graph.out_neighbors(i)) {
                if ((neighbor % mpi.num_processes) - mpi.rank == 0) {
                    if (utils::insert(translator, neighbor, num_vertices))
                        num_vertices++;
                    int to = neighbor;
                    // int to = translator[neighbor];
                    utils::insert(out_neighbors, from, to);
                    // utils::insert(in_neighbors, to, from);
                    num_edges++;
                }
            }
        }
        std::vector<int> assignment(graph.num_vertices(), -1);
        std::vector<int> myblocks;
        for (const std::pair<const int, int> &element : translator) {
            myblocks.push_back(element.first);
            assignment[element.first] = graph.assignment(element.first);
        }
        std::cout << "NOTE: rank " << mpi.rank << "/" << mpi.num_processes - 1 << " has N = " << num_vertices << " E = ";
        utils::print<int>(myblocks);
        std::cout << num_edges << std::endl;
        this->_num_blocks = num_vertices;
        this->_global_num_blocks = graph.num_vertices();
        this->_assignment = assignment;
        /// ================= TODO: move to partition() or distribute() method =====================
        // this->block_reduction_rate = block_reduction_rate;
        this->_overall_entropy = std::numeric_limits<float>::max();
        // this->blockmodel = std::make_unique<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
        // this->_blockmatrix = new DictMatrix(this->_num_blocks, this->_num_blocks);
        // Set the block assignment to be the range [0, this->num_blocks)
        // this->_blockmatrix = new DistDictMatrix(this->_global_num_blocks, this->_global_num_blocks, mpi, myblocks);
        // this->_ownership = std::vector<int>(graph.num_vertices(), -1);
        // Number of blocks to merge
        this->_num_blocks_to_merge = (int)(this->_num_blocks * BLOCK_REDUCTION_RATE);
        // this->sync_ownership(myblocks, mpi);
        // exit(-10000);
        // this->initialize_edge_counts(out_neighbors, mpi, myblocks);
        this->initialize_edge_counts(out_neighbors, myblocks);
    }
    /// TODO
    std::vector<int> build_mapping(std::vector<int> &values);
    /// Performs the block merges with the highest change in entropy/MDL
    void carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                               const std::vector<int> &best_merge_for_each_block);
    /// TODO implement this if needed
    // DistBlockmodel clone_with_true_block_membership(NeighborList &neighbors, std::vector<int> &true_block_membership);
    /// Returns a copy of the current Blockmodel
    DistBlockmodel copy();
    /// TODO implement this if needed
    // static DistBlockmodel from_sample(int num_blocks, NeighborList &neighbors, std::vector<int> &sample_block_membership,
    //                               std::map<int, int> &mapping, float block_reduction_rate);
    /// Uses the graph to fill out the blockmatrix and the block degree vectors.
    // void initialize_edge_counts(const NeighborList &neighbors, const MPI_Data &mpi, const std::vector<int> &myblocks);
    void initialize_edge_counts(const NeighborList &neighbors, const std::vector<int> &myblocks);
    /// TODO
    double log_posterior_probability();
    /// TODO
    void merge_blocks(int from_block, int to_block);
    /// TODO
    void move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                     std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                     std::vector<int> &new_block_degrees);
    /// TODO
    void set_block_membership(int vertex, int block);
    /// TODO
    void update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates);
    /// TODO: Get rid of getters and setters?
    std::vector<int> &assignment() { return this->_assignment; }
    int &assignment(int block) { return this->_assignment[block]; }
    void assignment(std::vector<int> block_assignment) { this->_assignment = block_assignment; }
    ISparseMatrix *blockmatrix() { return this->_blockmatrix; }
    std::vector<int> &degrees() { return this->_degrees; }
    int &degrees(int block) { return this->_degrees[block]; }
    void degrees(std::vector<int> block_degrees) { this->_degrees = block_degrees; }
    std::vector<int> &degrees_in() { return this->_degrees_in; }
    int &degrees_in(int block) { return this->_degrees_in[block]; }
    void degrees_in(std::vector<int> block_degrees_in) { this->_degrees_in = block_degrees_in; }
    std::vector<int> &degrees_out() { return this->_degrees_out; }
    int &degrees_out(int block) { return this->_degrees_out[block]; }
    void degrees_out(std::vector<int> block_degrees_out) { this->_degrees_out = block_degrees_out; }
    // float &getBlock_reduction_rate() { return this->block_reduction_rate; }
    // void setBlock_reduction_rate(float block_reduction_rate) { this->block_reduction_rate = block_reduction_rate; }
    int &num_blocks_to_merge() { return this->_num_blocks_to_merge; }
    void num_blocks_to_merge(int num) { this->_num_blocks_to_merge = num; }
    int &num_blocks() { return this->_num_blocks; }
    void num_blocks(int num) { this->_num_blocks = num; }
    float &overall_entropy() { return this->_overall_entropy; }
    void overall_entropy(float entropy) { this->_overall_entropy = entropy; }
    // std::vector<int> &ownership() { return this->_ownership; }
    // int &ownership(int block) { return this->_ownership[block]; }
    // void ownership(std::vector<int> new_ownership) { this->_ownership = new_ownership; }
    // Other
    bool empty;
    // TODO: method to sync after an iteration (rebalance blocks, etc)

  private:
    // Structure
    int _num_blocks;
    int _global_num_blocks;
    IDistSparseMatrix *_blockmatrix;
    // Known info
    std::vector<int> _assignment;
    std::vector<int> _degrees;
    std::vector<int> _degrees_in;
    std::vector<int> _degrees_out;
    // float block_reduction_rate;
    // Computed info
    float _overall_entropy;
    int _num_blocks_to_merge;
    /// Sorts the indices of an array in descending order according to the values of the array
    std::vector<int> sort_indices(const std::vector<double> &unsorted);
    // Distributed info
    /// The MPI_rank of the process that owns each block
    // std::vector<int> _ownership;
    /// Syncs the ownership between all MPI processes. NOTE: assumes _num_blocks and _global_num_blocks are correctly
    /// set already
    // void sync_ownership(const std::vector<int> &myblocks, const MPI_Data &mpi) {
    //     int numblocks[mpi.num_processes];
    //     MPI_Allgather(&(this->_num_blocks), 1, MPI_INT, &numblocks, 1, MPI_INT, MPI_COMM_WORLD);
    //     int offsets[mpi.num_processes];
    //     offsets[0] = 0;
    //     for (int i = 1; i < mpi.num_processes; ++i) {
    //         offsets[i] = offsets[i-1] + numblocks[i-1];
    //     }
    //     std::vector<int> allblocks(this->_global_num_blocks, -1);
    //     MPI_Allgatherv(myblocks.data(), this->_num_blocks, MPI_INT, allblocks.data(), &(numblocks[0]), &(offsets[0]), MPI_INT, MPI_COMM_WORLD);
    //     if (mpi.rank == 0) {
    //         utils::print<int>(allblocks);
    //     }
    //     int owner = 0;
    //     for (int i = 0; i < this->_global_num_blocks; ++i) {
    //         if (owner < mpi.num_processes - 1 && i >= offsets[owner+1]) {
    //             owner++;
    //         }
    //         this->_ownership[allblocks[i]] = owner;
    //     }
    //     if (mpi.rank == 0) {
    //         utils::print<int>(this->_ownership);
    //     }
    // }
};

#endif // SBP_DIST_BLOCKMODEL_HPP
