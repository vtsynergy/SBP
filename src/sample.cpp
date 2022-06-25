#include "sample.hpp"

#include <chrono>

#include "args.hpp"
#include "common.hpp"

namespace sample {

void es_add_vertex(const Graph &graph, ES_State &state, std::vector<int> &sampled, std::vector<int> &mapping,
                   int vertex) {
    sampled.push_back(vertex);
    int index = int(sampled.size()) - 1;
    mapping[vertex] = index;
    for (int neighbor : graph.out_neighbors(vertex)) {
        if (state.neighborhood_flag[neighbor]) continue;  // if already in neighborhood, ignore
        if (mapping[neighbor] >= 0) continue;  // if already sampled neighbor, ignore
        state.neighbors.insert(neighbor);
        state.neighborhood_flag[neighbor] = true;
        if (state.contribution[neighbor] > 0) continue;  // contribution has already been calculated
        es_update_contribution(graph, state, mapping, neighbor);  // this should also set contribution[vertex] to 0
    }
    state.neighbors.erase(vertex);
    state.neighborhood_flag[vertex] = false;
}

void es_update_contribution(const Graph &graph, ES_State &state, const std::vector<int> &mapping, int vertex) {
    for (int neighbor : graph.out_neighbors(vertex)) {
        if (state.neighborhood_flag[neighbor]) continue;
        if (mapping[neighbor] >= 0) continue;
        state.contribution[vertex]++;
        state.contribution_sum++;
    }
    for (int neighbor : graph.in_neighbors(vertex)) {
        if (state.contribution[neighbor] > 0) {
            state.contribution[neighbor]--;
            state.contribution_sum--;
        }
    }
}

Sample expansion_snowball(const Graph &graph) {
    std::vector<int> sampled;
    std::vector<int> mapping = utils::constant<int>(graph.num_vertices(), -1);
    ES_State state(graph.num_vertices());
    int start = common::random_integer(0, graph.num_vertices() - 1);
    es_add_vertex(graph, state, sampled, mapping, start);
    while (int(sampled.size()) < int(float(graph.num_vertices()) * args.samplesize)) {
        if (state.neighbors.empty()) {  // choose random vertex not already sampled
            int vertex;
            // Assuming sample size is < 50% (0.5), this should run less than 2 times on average.
            // If the graph consists of just one connected component, this whole if statement should never run at all.
            do {
                vertex = common::random_integer(0, graph.num_vertices() - 1);
            } while (mapping[vertex] >= 0);
            es_add_vertex(graph, state, sampled, mapping, vertex);
            continue;
        } else if (state.contribution_sum == 0) {  // choose random neighbor
            int index = common::random_integer(0, int(state.neighbors.size()) - 1);
            auto it = state.neighbors.begin();
            std::advance(it, index);
            int vertex = *it;
            es_add_vertex(graph, state, sampled, mapping, vertex);
            continue;
        }
        // choose neighbor with max contribution
        int vertex = utils::argmax<int>(state.contribution);
        es_add_vertex(graph, state, sampled, mapping, vertex);
    }
    return from_vertices(graph, sampled, mapping);
}

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
    return Blockmodel(sample_blockmodel.getNum_blocks(), graph, 0.5, assignment);
}

Sample from_vertices(const Graph &graph, const std::vector<int> &vertices, const std::vector<int> &mapping) {
    Graph sampled_graph(int(vertices.size()));
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
    sampled_graph.sort_vertices();
    return Sample { sampled_graph, mapping };
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
    return from_vertices(graph, sampled, mapping);
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
    return from_vertices(graph, sampled, mapping);
}

Sample sample(const Graph &graph) {
    if (args.samplingalg == "max_degree")
        return max_degree(graph);
    else if (args.samplingalg == "random")
        return random(graph);
    else if (args.samplingalg == "expansion_snowball")
        return expansion_snowball(graph);
    else
        return random(graph);
}

}