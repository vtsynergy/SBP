/***
 * Interface for storing a graph..
 */
#ifndef SBP_GRAPH_INTERFACE_HPP
#define SBP_GRAPH_INTERFACE_HPP

#include <iostream>
#include <string>

#include "blockmodel/sparse/typedefs.hpp"
#include "fs.hpp"
#include "utils.hpp"

// TODO: replace _out_neighbors and _in_neighbors with our DictTransposeMatrix
///
/// Graph interface.
///
class Graph {
public:
    virtual ~Graph() = default;
    //============================================
    // GETTERS & SETTERS
    //============================================
    /// Adds an edge to the graph
    virtual void add_edge(long from, long to) = 0;
    /// Returns a const reference to the assignment
    [[nodiscard]] const std::vector<long> &assignment() const { return this->_assignment; }
    /// Sets the assignment vector for the given graph
    void assignment(const std::vector<long> &assignment_vector) { this->_assignment = assignment_vector; }
    /// Returns the block/community assignment of vertex `v`
    [[nodiscard]] long assignment(long v) const { return this->_assignment[v]; }
    /// Sets the assignment of vertex `v` to block `b`
    void assign(long v, long b) { this->_assignment[v] = b; }
    /// Returns a vector containing the vertex degrees for every vertex in the graph
    [[nodiscard]] virtual std::vector<long> degrees() const = 0;
    /// Returns a const reference to the in neighbors
    [[nodiscard]] virtual const NeighborList &in_neighbors() const = 0;
    /// Returns a const reference to the in neighbors of vertex `v`
    [[nodiscard]] virtual const std::vector<long> &in_neighbors(long v) const  = 0;
    /// Returns the list of high degree vertices
    [[nodiscard]] const std::vector<long> &high_degree_vertices() const { return this->_high_degree_vertices; }
    /// Returns the list of low degree vertices
    [[nodiscard]] const std::vector<long> &low_degree_vertices() const { return this->_low_degree_vertices; }
    /// Calculates the modularity of this graph given a particular vertex-to-block `assignment`
    [[nodiscard]] double modularity(const std::vector<long> &assignment) const;
    /// Returns all the neighbors of a given vertex. Note that vertices that are both in- and out- neighbors are
    /// repeated.
    [[nodiscard]] virtual std::vector<long> neighbors(long vertex) const = 0;
    /// Returns the number of edges in this graph
    [[nodiscard]] long num_edges() const { return this->_num_edges; }
    /// Counts the number of island vertices in this graph
    [[nodiscard]] long num_islands() const;
    /// Returns the number of vertices in this graph
    [[nodiscard]] long num_vertices() const { return this->_num_vertices; }
    /// Returns a const reference to the out neighbors
    [[nodiscard]] const NeighborList &out_neighbors() const { return this->_out_neighbors; }
    /// Returns a const reference to the out neighbors of vertex `v`
    [[nodiscard]] const std::vector<long> &out_neighbors(long v) const { return this->_out_neighbors[v]; }
    /// Sorts the vertices into low and high degree vertices
    virtual void sort_vertices() = 0;
    /// Returns a list of edges, sorted by degree product
    [[nodiscard]] std::vector<std::pair<std::pair<long, long>, long>> sorted_edge_list() const;
    /// Sorts vertices into low and high influence vertices. Does this via vertex degree products of the graph edges
    void degree_product_sort();
protected:
    /// For every vertex, stores the community they belong to.
    /// If assignment[v] = -1, then the community of v is not known
    std::vector<long> _assignment;
    /// Stores true if vertex is one of the highest degree vertices
//    MapVector<bool> _high_degree_vertex;
    /// Stores a list of the high degree vertices
    std::vector<long> _high_degree_vertices;
    /// Stores a list of the low degree vertices
    std::vector<long> _low_degree_vertices;
    /// For every vertex, stores the incoming neighbors as a std::vector<long>
    NeighborList _in_neighbors;
    /// For every vertex, stores the outgoing neighbors as a std::vector<long>
    NeighborList _out_neighbors;
    /// The number of vertices in the graph
    long _num_vertices;
    /// The number of edges in the graph
    long _num_edges;
    /// Stores true if a vertex has self edges, false otherwise
    std::vector<bool> _self_edges;
};

#endif // SBP_GRAPH_INTERFACE_HPP
