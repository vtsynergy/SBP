#include "sample.hpp"

#include <chrono>

#include "args.hpp"
#include "common.hpp"
#include "mpi_data.hpp"

namespace sample {

Sample detach(const Graph &graph) {
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    std::vector<long> degrees = graph.degrees();
    long index = 0;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long degree = degrees[vertex];
        if (degree > 1) {
            sampled.push_back(vertex);
            mapping[vertex] = index;
            index++;
        }
    }
    return from_vertices(graph, sampled, mapping);
}

void es_add_vertex(const Graph &graph, ES_State &state, std::vector<long> &sampled, std::vector<long> &mapping,
                   long vertex) {
    sampled.push_back(vertex);
    long index = long(sampled.size()) - 1;
    mapping[vertex] = index;
    for (long neighbor : graph.out_neighbors(vertex)) {
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

void es_update_contribution(const Graph &graph, ES_State &state, const std::vector<long> &mapping, long vertex) {
    for (long neighbor : graph.out_neighbors(vertex)) {
        if (state.neighborhood_flag[neighbor]) continue;
        if (mapping[neighbor] >= 0) continue;
        state.contribution[vertex]++;
        state.contribution_sum++;
    }
    for (long neighbor : graph.in_neighbors(vertex)) {
        if (state.contribution[neighbor] > 0) {
            state.contribution[neighbor]--;
            state.contribution_sum--;
        }
    }
}

Sample expansion_snowball(const Graph &graph) {
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    ES_State state(graph.num_vertices());
    long start = common::random_longeger(0, graph.num_vertices() - 1);
    es_add_vertex(graph, state, sampled, mapping, start);
    while (long(sampled.size()) < long(double(graph.num_vertices()) * args.samplesize)) {
        if (state.neighbors.empty()) {  // choose random vertex not already sampled
            long vertex;
            // Assuming sample size is < 50% (0.5), this should run less than 2 times on average.
            // If the graph consists of just one connected component, this whole if statement should never run at all.
            do {
                vertex = common::random_longeger(0, graph.num_vertices() - 1);
            } while (mapping[vertex] >= 0);
            es_add_vertex(graph, state, sampled, mapping, vertex);
            continue;
        } else if (state.contribution_sum == 0) {  // choose random neighbor
            long index = common::random_longeger(0, long(state.neighbors.size()) - 1);
            auto it = state.neighbors.begin();
            std::advance(it, index);
            long vertex = *it;
            es_add_vertex(graph, state, sampled, mapping, vertex);
            continue;
        }
        // choose neighbor with max contribution
        long vertex = utils::argmax<long>(state.contribution);
        es_add_vertex(graph, state, sampled, mapping, vertex);
    }
    return from_vertices(graph, sampled, mapping);
}

Blockmodel extend(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample) {
    std::cout << "Extending the sample results to the full graph" << std::endl;
    std::vector<long> assignment = utils::constant<long>(graph.num_vertices(), -1);
    // Embed the known assignments from the partitioned sample
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long sample_vertex = sample.mapping[vertex];
        if (sample_vertex == -1) continue;
        assignment[vertex] = sample_blockmodel.block_assignment(sample_vertex);
    }
    // Infer membership of remaining vertices
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (assignment[vertex] != -1) continue;  // already assigned
        // Count edges to/from different communities
        MapVector<long> edge_counts;
        for (long neighbor : graph.out_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1) continue;  // we don't know neighbor's community
            edge_counts[community]++;
        }
        for (long neighbor : graph.in_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1 || neighbor == vertex) continue;
            edge_counts[community]++;
        }
        if (edge_counts.empty()) {  // assign random community
            long community = common::random_longeger(0, sample_blockmodel.getNum_blocks() - 1);
            assignment[vertex] = community;
            continue;
        }
        long max_edges = 0;
        long likely_community = -1;
        for (const auto &element : edge_counts) {
            long community = element.first;
            long edges = element.second;
            if (edges > max_edges) {
                max_edges = edges;
                likely_community = community;
            }
        }
        assignment[vertex] = likely_community;
    }
    return Blockmodel(sample_blockmodel.getNum_blocks(), graph, 0.5, assignment);
}

Sample from_vertices(const Graph &graph, const std::vector<long> &vertices, const std::vector<long> &mapping) {
    Graph sampled_graph(long(vertices.size()));
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long vertex_id = mapping[vertex];
        if (vertex_id == -1) continue;
        const std::vector<long> &neighbors = graph.out_neighbors(vertex);
        for (long neighbor : neighbors) {
            long neighbor_id = mapping[neighbor];
            if (neighbor_id == -1) continue;
            sampled_graph.add_edge(vertex_id, neighbor_id);
        }
        sampled_graph.assign(vertex_id, graph.assignment(vertex));
    }
    sampled_graph.sort_vertices();
    return Sample { sampled_graph, mapping };
}

Sample max_degree(const Graph &graph) {
    std::vector<long> vertex_degrees = graph.degrees();
    std::vector<long> indices = utils::argsort(vertex_degrees);
//    std::vector<long> indices = utils::range<long>(0, graph.num_vertices());
//    std::sort(indices.data(), indices.data() + indices.size(),  // sort in descending order
//              [vertex_degrees](size_t i1, size_t i2) { return vertex_degrees[i1] > vertex_degrees[i2]; });
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    for (long index = 0; index < long(args.samplesize * double(graph.num_vertices())); ++index) {
        long vertex = indices[index];
        sampled.push_back(vertex);
        mapping[vertex] = index;  // from full graph ID to sample graph ID
    }
    return from_vertices(graph, sampled, mapping);
}

Sample random(const Graph &graph) {
    std::vector<long> indices = utils::range<long>(0, graph.num_vertices());
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::mt19937_64(seed));
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    for (long index = 0; index < long(args.samplesize * double(graph.num_vertices())); ++index) {
        long vertex = indices[index];
        sampled.push_back(vertex);
        mapping[vertex] = index;  // from full graph ID to sample graph ID
    }
    return from_vertices(graph, sampled, mapping);
}

Sample round_robin(const Graph &graph, int subgraph_index, int num_subgraphs) {
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    long index = 0;
    for (long vertex = subgraph_index; vertex < graph.num_vertices(); vertex += num_subgraphs) {
        sampled.push_back(vertex);
	mapping[vertex] = index;
	index++;
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

Blockmodel reattach(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample) {
    std::cout << "Extending the sample results to the full graph with size: " << graph.num_vertices() << std::endl;
    std::vector<long> assignment = utils::constant<long>(graph.num_vertices(), -1);
    // Embed the known assignments from the partitioned sample
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long sample_vertex = sample.mapping[vertex];
        if (sample_vertex == -1) continue;
        assignment[vertex] = sample_blockmodel.block_assignment(sample_vertex);
    }
    // Infer membership of remaining vertices
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (assignment[vertex] != -1) continue;  // already assigned
        long random_community = common::random_longeger(0, sample_blockmodel.getNum_blocks() - 1);
        // Assign to the same community as only neighbor
        for (long neighbor : graph.out_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1) {
                assignment[vertex] = random_community;
                assignment[neighbor] = random_community;
                break;
            }
            assignment[vertex] = community;
        }
        for (long neighbor : graph.in_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1) {
                assignment[vertex] = random_community;
                assignment[neighbor] = random_community;
                break;
            }
            assignment[vertex] = community;
        }
        // Vertex is an island
        if (assignment[vertex] < 0) {  // assign random community
            assignment[vertex] = random_community;
        }
    }
    return { sample_blockmodel.getNum_blocks(), graph, 0.5, assignment };
}

}
