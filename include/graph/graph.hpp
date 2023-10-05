//
// Created by Frank on 10/3/2023.
//

#ifndef SBP_GRAPH_HPP
#define SBP_GRAPH_HPP

#include <iostream>
#include <vector>

#include "graph/graph_interface.hpp"

namespace graph {

/// Loads the graph.
/// Assumes the graph file is named: <args.filepath>.tsv
/// Assumes the true assignment file is named: <args.filepath>_truePartition.tsv
Graph* load();

/// Loads the graph if it's in a matrix market format.
Graph* load_matrix_market(std::vector<std::vector<std::string>> &csv_contents);

/// Loads the graph if it's in a text format: a list of "from to" string pairs.
Graph* load_text(std::vector<std::vector<std::string>> &csv_contents);

/// Creates a new empty graph with `num_vertices` vertices.
Graph* make_graph(long num_vertices);

/// Parses a directed graph from csv contents
void parse_directed(NeighborList &out_neighbors, NeighborList &in_neighbors, long &num_vertices,
                    std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents);

/// Parses an undirected graph from csv contents
void parse_undirected(NeighborList &neighbors, long &num_vertices, std::vector<bool> &self_edges,
                      std::vector<std::vector<std::string>> &contents);

Graph* package(NeighborList &out_neighbors, NeighborList &in_neighbors, long &num_vertices, long& num_edges,
               std::vector<bool> &self_edges);

}  // namespace graph

#endif // SBP_GRAPH_HPP
