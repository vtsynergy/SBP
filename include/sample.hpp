/**
* Contains code for sampling from graphs.
*/
#ifndef SBP_SAMPLE_HPP
#define SBP_SAMPLE_HPP

#include <vector>

#include "blockmodel/blockmodel.hpp"
#include "graph.hpp"

namespace sample {

struct Sample {
    Graph graph;
    std::vector<int> mapping;
};

/// Samples vertices using the expansion snowball algorithm of Maiya et al.
Sample expansion_snowball(const Graph &graph);

/// Extends the results from the sample graph blockmodel to the full graph blockmodel.
Blockmodel extend(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample);

/// Samples vertices with the highest degrees.
Sample max_degree(const Graph &graph);

/// Samples random vertices.
Sample random(const Graph &graph);

}

#endif // SBP_SAMPLE_HPP
