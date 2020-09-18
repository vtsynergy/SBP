/***
 * Stores a Graph.
 */
#ifndef SBP_EVALUATE_HPP
#define SBP_EVALUATE_HPP

#include <set>

#include "hungarian.hpp"

#include "graph.hpp"
#include "partition/partition.hpp"

namespace evaluate {

double calculate_f1_score(const Graph &graph, Hungarian::Matrix &contingency_table);

void evaluate_partition(const Graph &graph, Partition &partition);

Hungarian::Matrix hungarian(const Graph &graph, Partition &partition);

}

#endif // SBP_EVALUATE_HPP
