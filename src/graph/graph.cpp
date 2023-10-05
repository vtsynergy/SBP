//
// Created by Frank on 10/3/2023.
//

#include "graph/graph.hpp"

#include "graph/directed_graph.hpp"
#include "graph/undirected_graph.hpp"
#include "mpi_data.hpp"

namespace graph {

Graph* load() {
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string base_path = utils::build_filepath();
    fs::path graph_path = base_path + ".tsv";
    fs::path truth_path = base_path + "_truePartition.tsv";
    // TODO: Handle weighted graphs
    std::vector<std::vector<std::string>> csv_contents = utils::read_csv(graph_path);
    if (csv_contents.empty()) {
        graph_path = base_path + ".mtx";
        csv_contents = utils::read_csv(graph_path);
    }
    Graph* graph;
    if (csv_contents[0][0] == "%%MatrixMarket") {
        graph = load_matrix_market(csv_contents);
    } else {
        graph = load_text(csv_contents);
    }
    if (mpi.rank == 0)
        std::cout << "V: " << graph->num_vertices() << " E: " << graph->num_edges() << std::endl;

    csv_contents = utils::read_csv(truth_path);
    std::vector<long> assignment;
    // TODO: vertices, communities should be size_t or ulong. Will need to make sure -1 returns are properly handled
    // elsewhere.
    if (!csv_contents.empty()) {
        for (std::vector<std::string> &assign: csv_contents) {
            long vertex = std::stoi(assign[0]) - 1;
            long community = std::stoi(assign[1]) - 1;
            if (vertex >= (long)assignment.size()) {
                std::vector<long> padding(vertex - assignment.size() + 1, -1);
                assignment.insert(assignment.end(), padding.begin(), padding.end());
            }
            assignment[vertex] = community;
        }
    } else {
        assignment = utils::constant<long>(graph->num_vertices(), 0);
    }
    graph->assignment(assignment);
    return graph;
}

Graph* make_graph(long num_vertices) {
    if (args.undirected)
        return new UndirectedGraph(num_vertices);
    return new DirectedGraph(num_vertices);
}

/// Loads the graph if it's in a matrix market format.
Graph* load_matrix_market(std::vector<std::vector<std::string>> &csv_contents) {
    // TODO: properly handle undirected matrices
    if (csv_contents[0][2] != "coordinate") {
        std::cerr << "ERROR " << "Dense matrices are not supported!" << std::endl;
        exit(-1);
    }
    if (csv_contents[0][4] == "symmetric") {
        std::cout << "Graph is symmetric" << std::endl;
        args.undirected = true;
    }
    // Find index at which edges start
    long index = 0;
    long num_vertices, num_edges;
    for (long i = 0; i < (long) csv_contents.size(); ++i) {
        const std::vector<std::string> &line = csv_contents[i];
//        std::cout << "line: ";
//        utils::print<std::string>(line);
        if (line[0][0] == '%') continue;
        num_vertices = std::stoi(line[0]);
        if (num_vertices != std::stoi(line[1])) {
            std::cerr << "ERROR " << "Rectangular matrices are not supported!" << std::endl;
            exit(-1);
        }
        num_edges = std::stoi(line[2]);
        index = i + 1;
        break;
    }
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges = utils::constant<bool>(num_vertices, false);
    for (long i = index; i < (long) csv_contents.size(); ++i) {
        const std::vector<std::string> &edge = csv_contents[i];
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to , from);
        if (args.undirected && from != to) {  // Force symmetric graph to be directed by including reverse edges.
            utils::insert_nodup(out_neighbors, to, from);
            utils::insert_nodup(in_neighbors, from , to);
//            num_edges++;
        }
        if (from == to) {
            self_edges[from] = true;
        }
//        num_edges++;
    }
    // Pad the neighbors lists
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
    return package(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
}

/// Loads the graph if it's in a text format: a list of "from to" string pairs.
Graph* load_text(std::vector<std::vector<std::string>> &csv_contents) {
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges;
    long num_vertices = 0;
    if (args.undirected)
        parse_undirected(out_neighbors, num_vertices, self_edges, csv_contents);
    else
        parse_directed(out_neighbors, in_neighbors, num_vertices, self_edges, csv_contents);
    long num_edges = 0;  // TODO: unnecessary re-counting of edges?
    for (const std::vector<long> &neighborhood : out_neighbors) {
        num_edges += (long)neighborhood.size();
    }
    if (args.undirected) {
        num_edges /= 2;
    }
    return package(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
}

void parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, long &num_vertices,
                    std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to, from);
        while (long(self_edges.size()) < num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
}

void parse_undirected(NeighborList &neighbors, long &num_vertices, std::vector<bool> &self_edges,
                      std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(neighbors, from , to);
        if (from != to)
            utils::insert_nodup(neighbors, to, from);
        while (long(self_edges.size()) < num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    while (neighbors.size() < size_t(num_vertices)) {
        neighbors.push_back(std::vector<long>());
    }
}

Graph* package(NeighborList &out_neighbors, NeighborList &in_neighbors, long &num_vertices, long& num_edges,
               std::vector<bool> &self_edges) {
    Graph* graph;
    if (args.undirected)
        graph = new UndirectedGraph(out_neighbors, num_vertices, num_edges, self_edges);
    else
        graph = new DirectedGraph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
    return graph;
}

}  // namespace graph
