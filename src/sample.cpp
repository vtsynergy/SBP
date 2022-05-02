#include "sample.hpp"

#include <chrono>

#include "args.hpp"
#include "common.hpp"

namespace sample {

Blockmodel extend(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample) {
    std::cout << "Extending the sample results to the full graph" << std::endl;
    std::vector<int> assignment = utils::constant<int>(graph.num_vertices(), -1);
    // Embed the known assignments from the partitioned sample
    for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        int sample_vertex = sample.mapping[vertex];
        if (sample_vertex == -1) continue;
        assignment[vertex] = sample_blockmodel.block_assignment(sample_vertex);
    }
    // Infer membership of remaining vertices
    for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (assignment[vertex] != -1) continue;  // already assigned
        // Count edges to/from different communities
        MapVector<int> edge_counts;
        for (int neighbor : graph.out_neighbors(vertex)) {
            int community = assignment[neighbor];
            if (community == -1) continue;  // we don't know neighbor's community
            edge_counts[community]++;
        }
        for (int neighbor : graph.in_neighbors(vertex)) {
            int community = assignment[neighbor];
            if (community == -1 || neighbor == vertex) continue;
            edge_counts[community]++;
        }
        if (edge_counts.empty()) {  // assign random community
            int community = common::random_integer(0, sample_blockmodel.getNum_blocks() - 1);
            assignment[vertex] = community;
            continue;
        }
        int max_edges = 0;
        int likely_community = -1;
        for (const auto &element : edge_counts) {
            int community = element.first;
            int edges = element.second;
            if (edges > max_edges) {
                max_edges = edges;
                likely_community = community;
            }
        }
        assignment[vertex] = likely_community;
    }
    return Blockmodel(sample_blockmodel.getNum_blocks(), graph.out_neighbors(), 0.5, assignment);
}

Sample max_degree(const Graph &graph) {
    std::vector<int> vertex_degrees = graph.degrees();
    std::vector<int> indices = utils::range<int>(0, graph.num_vertices());
    std::sort(indices.data(), indices.data() + indices.size(),  // sort in descending order
              [vertex_degrees](size_t i1, size_t i2) { return vertex_degrees[i1] > vertex_degrees[i2]; });
    std::vector<int> sampled;
    std::vector<int> mapping = utils::constant(graph.num_vertices(), -1);
    for (int index = 0; index < int(args.samplesize * float(graph.num_vertices())); ++index) {
        int vertex = indices[index];
        sampled.push_back(vertex);
        mapping[vertex] = index;  // from full graph ID to sample graph ID
    }
    Graph sampled_graph(int(sampled.size()));
    for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        int vertex_id = mapping[vertex];
        if (vertex_id == -1) continue;
        const std::vector<int> &neighbors = graph.out_neighbors(vertex);
        for (int neighbor : neighbors) {
            int neighbor_id = mapping[neighbor];
            if (neighbor_id == -1) continue;
            sampled_graph.add_edge(vertex_id, neighbor_id);
        }
        sampled_graph.assign(vertex_id, graph.assignment(vertex));
    }
    return Sample { sampled_graph, mapping };
}

Sample random(const Graph &graph) {
    std::vector<int> indices = utils::range<int>(0, graph.num_vertices());
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::mt19937_64(seed));
    std::vector<int> sampled;
    std::vector<int> mapping = utils::constant(graph.num_vertices(), -1);
    for (int index = 0; index < int(args.samplesize * float(graph.num_vertices())); ++index) {
        int vertex = indices[index];
        sampled.push_back(vertex);
        mapping[vertex] = index;  // from full graph ID to sample graph ID
    }
    Graph sampled_graph(int(sampled.size()));
    for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        int vertex_id = mapping[vertex];
        if (vertex_id == -1) continue;
        const std::vector<int> &neighbors = graph.out_neighbors(vertex);
        for (int neighbor: neighbors) {
            int neighbor_id = mapping[neighbor];
            if (neighbor_id == -1) continue;
            sampled_graph.add_edge(vertex_id, neighbor_id);
        }
        sampled_graph.assign(vertex_id, graph.assignment(vertex));
    }
    return Sample{sampled_graph, mapping};
}
}