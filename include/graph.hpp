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

class Graph {
public:
    Graph(NeighborList &out_neighbors, NeighborList &in_neighbors, int num_vertices, int num_edges,
          const std::vector<int> &assignment = std::vector<int>()) {
        this->out_neighbors = out_neighbors;
        this->in_neighbors = in_neighbors;
        this->num_vertices = num_vertices;
        this->num_edges = num_edges;
        this->assignment = assignment;
    }
    Graph() = default;
    /// Loads the graph. Assumes the file is saved in the following directory:
    /// <args.directory>/<args.type>/<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar
    /// Assumes the graph file is named:
    /// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_nodes.tsv
    /// Assumes the true assignment file is named:
    /// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_trueBlockmodel.tsv
    static Graph load(Args &args);
// private:
    /// For every vertex, stores the outgoing neighbors as a std::vector<int>
    NeighborList out_neighbors;
    /// For every vertex, stores the incoming neighbors as a std::vector<int>
    NeighborList in_neighbors;
    /// The number of vertices in the graph
    int num_vertices;
    /// The number of edges in the graph
    int num_edges;
    /// For every vertex, stores the community they belong to.
    /// If assignment[v] = -1, then the community of v is not known
    std::vector<int> assignment;
};

#endif // SBP_GRAPH_HPP
