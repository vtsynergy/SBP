#include "sample.hpp"

#include "args.hpp"

namespace sample {

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
    // TODO: re-map the assignments
    return Sample { sampled_graph, mapping };
}

}