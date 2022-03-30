/***
 * Stores a Graph.
 */
#ifndef SBP_GRAPH_HPP
#define SBP_GRAPH_HPP

// #include <filesystem>
// #include <fstream>
#include <iostream>
// #include <sstream>
#include <string>
// #include <limits>

// #include <Eigen/Core>

// #include "argparse/argparse.hpp"
#include "args.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "fs.hpp"
#include "utils.hpp"

// #include "sparse/boost_mapped_matrix.hpp"

// typedef py::EigenDRef<Eigen::Matrix<int, Eigen::Dynamic, 2>> Matrix2Column;

// typedef std::vector<std::vector<int>> VarLengthMatrix;

// TODO: replace _out_neighbors and _in_neighbors with our DictTransposeMatrix
class Graph {
public:
    // TODO: figure out how to get rid of in_neighbors for undirected graphs (save some space)
    Graph(NeighborList &out_neighbors, NeighborList &in_neighbors, int num_vertices, int num_edges,
          const std::vector<int> &assignment = std::vector<int>()) {
        this->_out_neighbors = out_neighbors;
        this->_in_neighbors = in_neighbors;
        this->_num_vertices = num_vertices;
        this->_num_edges = num_edges;
        this->_assignment = assignment;
    }
    Graph() = default;
    /// Loads the graph. Assumes the file is saved in the following directory:
    /// <args.directory>/<args.type>/<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar
    /// Assumes the graph file is named:
    /// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_nodes.tsv
    /// Assumes the true assignment file is named:
    /// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_trueBlockmodel.tsv
    static Graph load(Args &args);
    //============================================
    // GETTERS & SETTERS
    //============================================
    /// Returns a const reference to the assignmnet
    const std::vector<int> &assignment() const { return this->_assignment; }
    /// Returns the block/community assignment of vertex `v`
    int assignment(int v) const { return this->_assignment[v]; }
    /// Sets the assignment of vertex `v` to block `b`
    void assign(int v, int b) { this->_assignment[v] = b; }
    /// Returns a const reference to the in neighbors
    const NeighborList &in_neighbors() const { return this->_in_neighbors; }
    /// Returns a const reference to the in neighbors of vertex `v`
    const std::vector<int> &in_neighbors(int v) const { return this->_in_neighbors[v]; }
    /// Calculates the modularity of this graph given a particular vertex-to-block `assignment`
    double modularity(const std::vector<int> &assignment) const;
    /// Returns the number of edges in this graph
    int num_edges() const { return this->_num_edges; }
    /// Returns the number of vertices in this graph
    int num_vertices() const { return this->_num_vertices; }
    /// Returns a const reference to the out neighbors
    const NeighborList &out_neighbors() const { return this->_out_neighbors; }
    /// Returns a const reference to the out neighbors of vertex `v`
    const std::vector<int> &out_neighbors(int v) const { return this->_out_neighbors[v]; }
private:
    /// For every vertex, stores the outgoing neighbors as a std::vector<int>
    NeighborList _out_neighbors;
    /// For every vertex, stores the incoming neighbors as a std::vector<int>
    NeighborList _in_neighbors;
    /// The number of vertices in the graph
    int _num_vertices;
    /// The number of edges in the graph
    int _num_edges;
    /// For every vertex, stores the community they belong to.
    /// If assignment[v] = -1, then the community of v is not known
    std::vector<int> _assignment;
    /// Parses a directed graph from csv contents
    static void parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                               std::vector<std::vector<std::string>> &contents);
    /// Parses an undirected graph from csv contents
    static void parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                                 std::vector<std::vector<std::string>> &contents);
};

#endif // SBP_GRAPH_HPP
