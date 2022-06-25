/**
* Contains code for sampling from graphs.
*/
#ifndef SBP_SAMPLE_HPP
#define SBP_SAMPLE_HPP

#include <vector>

#include "blockmodel/blockmodel.hpp"
#include "graph.hpp"

namespace sample {

struct ES_State {
    ES_State(int num_vertices) {
        this->contribution = utils::constant<int>(num_vertices, 0);
        this->neighborhood_flag = utils::constant<bool>(num_vertices, false);
        this->neighbors = std::set<int>();
        this->contribution_sum = 0;
    }
    std::vector<int> contribution;
    std::vector<bool> neighborhood_flag;
    std::set<int> neighbors;
    int contribution_sum;
};

struct Sample {
    Graph graph;
    std::vector<int> mapping;
};

/// Adds `vertex` to the expansion snowball sample.
void es_add_vertex(const Graph &graph, ES_State &state, std::vector<int> &sampled, std::vector<int> &mapping,
                   int vertex);

/// Updates the contribution of `vertex`, which has just been placed in the neighborhood of the current sample.
/// Also decreases the contribution of all vertices that link to `vertex`.
void es_update_contribution(const Graph &graph, ES_State &state, const std::vector<int> &mapping, int vertex);

/// Samples vertices using the expansion snowball algorithm of Maiya et al.
Sample expansion_snowball(const Graph &graph);

/// Extends the results from the sample graph blockmodel to the full graph blockmodel.
Blockmodel extend(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample);

/// Creates a Sample from sampled vertices and their mappings.
Sample from_vertices(const Graph &graph, const std::vector<int> &vertices, const std::vector<int> &mapping);

/// Samples vertices with the highest degrees.
Sample max_degree(const Graph &graph);

/// Samples random vertices.
Sample random(const Graph &graph);

/// Creates a sample using args.samplingalg algorithm
Sample sample(const Graph &graph);

}

#endif // SBP_SAMPLE_HPP
