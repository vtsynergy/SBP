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
#include <random>
#include <queue>

#include "../args.hpp"
#include "blockmodel.hpp"
#include "../graph.hpp"
#include "mpi_data.hpp"
#include "sparse/dict_matrix.hpp"
#include "sparse/dict_transpose_matrix.hpp"
#include "sparse/typedefs.hpp"
#include "../utils.hpp"


class TwoHopBlockmodel : public Blockmodel {
public:
    // using Blockmodel::Blockmodel;
    // Constructors are not derived from base class
    TwoHopBlockmodel() : Blockmodel() {}
    TwoHopBlockmodel(int num_blocks, float block_reduction_rate) : Blockmodel(num_blocks, block_reduction_rate) {}
    TwoHopBlockmodel(int num_blocks, const NeighborList &out_neighbors, float block_reduction_rate)
            : TwoHopBlockmodel(num_blocks, block_reduction_rate) {
        // If the block assignment is not provided, use round-robin assignment
        this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
        for (int i = mpi.rank; i < this->num_blocks; i += mpi.num_processes) {  // round-robin work mapping
            this->_my_blocks[i] = true;
        }
        this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);  // no distribution
        this->initialize_edge_counts(out_neighbors);
    }
    TwoHopBlockmodel(int num_blocks, const NeighborList &out_neighbors, float block_reduction_rate,
                     std::vector<int> &block_assignment) : TwoHopBlockmodel(num_blocks, block_reduction_rate) {
        // Set the block assignment
        this->_block_assignment = block_assignment;
        this->distribute(out_neighbors);
        this->initialize_edge_counts(out_neighbors);
    }
    /// Sets the _in_two_hop_radius for a 2-hop blockmodel.
    void build_two_hop_blockmodel(const NeighborList &neighbors);
    TwoHopBlockmodel copy();
    /// Distributes the blockmodel amongst MPI ranks. Needs to be called before the first call to
    /// initialize_edge_counts, since it sets the _in_two_hop_radius and _my_blocks vectors. After that, it only needs
    /// to be called to re-distribute the blockmodel (followed by initialize_edge_counts).
    void distribute(const NeighborList &neighbors);
    /// Returns the _in_two_hop_radius vector.
    const std::vector<bool>& in_two_hop_radius() const { return this->_in_two_hop_radius; }
    void initialize_edge_counts(const NeighborList &neighbors);
    double log_posterior_probability() const;
    /// Returns true if this blockmodel owns the compute for the requested block.
    bool owns_block(int block) const;
    /// Returns true if this blockmodel owns the compute for the requested vertex.
    bool owns_vertex(int vertex) const;
    /// Returns true if this blockmodel owns storage for the requested block.
    bool stores(int block) const;
private:
    // ===== Functions
    /// Returns a sorted vector of <block, block size> pairs, in descending order of block size.
    std::vector<std::pair<int,int>> sorted_block_sizes() const;
    /// No data distribution, work is mapped using round-robin strategy.
    void distribute_none();
    /// 2-Hop data distribution using round-robin assignment, each MPI rank responsible for the vertices in the blocks
    /// mapped to it.
    void distribute_2hop_round_robin(const NeighborList &neighbors);
    /// 2-Hop data distribution, balanced by block size, each MPI rank responsible for the vertices in the blocks
    /// mapped to it.
    void distribute_2hop_size_balanced(const NeighborList &neighbors);
    /// 2-Hop data distribution, based on snowball sampling over vertices, each MPI rank responsible for the vertices
    /// in the blocks mapped to it.
    void distribute_2hop_snowball(const NeighborList &neighbors);
    // ===== Variables
    /// Stores true for in_two_hop_radius[block] if block is stored in this blockmodel.
    std::vector<bool> _in_two_hop_radius;
    /// Stores true for my_blocks[block] if this blockmodel owns the compute for this block.
    std::vector<bool> _my_blocks;
    /// Stores 1 for any vertex that this blockmodel owns.
    std::vector<int> _my_vertices;
};

#endif // SBP_DIST_BLOCKMODEL_HPP