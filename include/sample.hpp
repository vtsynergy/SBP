/**
* Contains code for sampling from graphs.
*/
#ifndef SBP_SAMPLE_HPP
#define SBP_SAMPLE_HPP

#include <vector>

#include "graph.hpp"

namespace sample {

struct Sample {
    Graph graph;
    std::vector<int> mapping;
};

/// Samples vertices with the highest degrees.
Sample max_degree(const Graph &graph);

/// Samples vertices using the expansion snowball algorithm of Maiya et al.
Sample expansion_snowball(const Graph &graph);

}

#endif // SBP_SAMPLE_HPP
