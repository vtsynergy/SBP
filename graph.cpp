#include "graph.hpp"
#include "utils.hpp"
#include "mpi_data.hpp"

Graph Graph::load(Args &args) {
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string basepath = utils::build_filepath(args);
    fs::path graphpath = basepath + ".tsv";
    fs::path truthpath = basepath + "_truePartition.tsv";
    // TODO: Handle weighted graphs
    std::vector<std::vector<std::string>> csv_contents = utils::read_csv(graphpath);
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    int num_edges = csv_contents.size();
    int num_vertices = 0;
    if (args.undirected)
        Graph::parse_undirected(in_neighbors, out_neighbors, num_vertices, csv_contents);
    else
        Graph::parse_directed(in_neighbors, out_neighbors, num_vertices, csv_contents);
    // std::cout << "num_edges before: " << num_edges << std::endl;
    num_edges = 0;  // TODO: unnecessary re-counting of edges?
    for (const std::vector<int> &neighborhood : out_neighbors) {
        for (int neighbor : neighborhood) {
            num_edges++;
        }
    }
    if (args.undirected) {
        num_edges /= 2;
    }
    // std::cout << "num_edges after: " << num_edges << std::endl;
    if (mpi.rank == 0)
        std::cout << "V: " << num_vertices << " E: " << num_edges << std::endl;

    csv_contents = utils::read_csv(truthpath);
    std::vector<int> assignment;
    if (!csv_contents.empty()) {
        for (std::vector<std::string> &assign: csv_contents) {
            int vertex = std::stoi(assign[0]) - 1;
            int community = std::stoi(assign[1]) - 1;
            if (vertex >= assignment.size()) {
                std::vector<int> padding(vertex - assignment.size() + 1, -1);
                assignment.insert(assignment.end(), padding.begin(), padding.end());
            }
            assignment[vertex] = community;
        }
    }
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
}

void Graph::parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                           std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to, from);
    }
}

void Graph::parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                             std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        if (from != to)
            utils::insert_nodup(out_neighbors, to, from);
        in_neighbors = NeighborList(out_neighbors);
    }
}
