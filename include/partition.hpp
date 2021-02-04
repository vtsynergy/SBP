/**
 * Partition the graph amongst multiple MPI processes
 */
#ifndef SBP_PARTITION_HPP
#define SBP_PARTITION_HPP

#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

#include "args.hpp"
#include "blockmodel.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "graph.hpp"
#include "mpi_utils.hpp"

namespace partition {

/// Represents a single partition of a distributed blockmodel.
class BlockmodelPartition {
public:
    BlockmodelPartition(std::vector<int> &communities, int global_num_blocks, const NeighborList &out_neighbors,
                        int BLOCK_REDUCTION_RATE) {
        this->local_num_blocks_ = communities.size();
        this->communities_ = communities;
        this->community_flag_ = std::vector<bool>(global_num_blocks, false);
        for (int community : communities)
            this->community_flag_[community] = true;
        this->blockmodel_ = Blockmodel(global_num_blocks, out_neighbors, BLOCK_REDUCTION_RATE);
    }
    Blockmodel &blockmodel() { return this->blockmodel_; }
    std::vector<int> &communities() { return this->communities_; }
    /// Returns true if the partitions is responsible for `community`.
    bool contains(int community);
    /// The number of communities stored locally
    int local_num_blocks() { return this->local_num_blocks_; }
private:
    Blockmodel blockmodel_;
    std::vector<int> communities_;  // The communities this partition is responsible for
    std::vector<bool> community_flag_;
    int local_num_blocks_;
};

/// Represents a single partition of a distributed graph.
class GraphPartition {
public:
    GraphPartition(int global_num_vertices, int global_num_edges, std::vector<int> &vertices, 
                   NeighborList &out_neighbors, NeighborList &in_neighbors, int num_vertices, int num_edges,
                   const std::vector<int> &assignment = std::vector<int>()) {
        this->global_num_vertices_ = global_num_vertices;
        this->global_num_edges_ = global_num_edges;
        std::cout << "About to create le graph" << std::endl;
        this->graph_ = Graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
        std::cout << "Created le graph" << std::endl;
        this->vertices_ = vertices;
    }
    int global_num_vertices() { return this->global_num_vertices_; }
    int global_num_edges() { return this->global_num_edges_; }
    const Graph &graph() { return this->graph_; }
    std::vector<int> &vertices() { return this->vertices_; }  // Maybe TODO: add a const version of this method
    void vertices(std::vector<int> &vertices) { this->vertices_ = vertices; }
private:
    int global_num_vertices_;
    int global_num_edges_;
    Graph graph_;
    std::vector<int> vertices_;  // The vertices this partition is responsible for
};

/// Distributes the graph using the round-robin method. Similar to `partition_round_robin`, but not does change the
/// vertex indices.
GraphPartition distribute(const Graph &graph, utils::mpi::Info &mpi, Args &args);

/// Partitions the graph using the method chosen via `args`.
Graph partition(const Graph &graph, int rank, int num_processes, Args &args);

/// Partitions the graph using the round robin strategy. The resulting partitions do not overlap.
Graph partition_round_robin(const Graph &graph, int rank, int num_processes, int target_num_vertices);

/// Partitions the graph using the random strategy. The resulting partitions do not overlap.
Graph partition_random(const Graph &graph, int rank, int num_processes, int target_num_vertices);

/// Partitions the graph using the snowball sampling strategy. The resulting partitions DO overlap.
Graph partition_snowball(const Graph &graph, int rank, int num_processes, int target_num_vertices);

} // namespace partition

#endif // SBP_PARTITION_HPP