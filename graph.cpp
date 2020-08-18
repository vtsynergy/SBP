#include "graph.hpp"
#include "utils.hpp"

Graph Graph::load(argparse::ArgumentParser &args) {
    // TODO: Add capability to process multiple "streaming" grpah parts
    std::string basepath = utils::build_filepath(args);
    std::filesystem::path graphpath = basepath + ".tsv";
    std::filesystem::path truthpath = basepath + "_truePartition.tsv";
    // TODO: Handle weighted graphs
    std::vector<std::vector<std::string>> csv_contents = utils::read_csv(graphpath);
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    int num_edges = csv_contents.size();
    int num_vertices = 0;
    for (std::vector<std::string> &edge : csv_contents) {
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indeces vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert(out_neighbors, from , to);
        utils::insert(in_neighbors, to, from);
    }
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
