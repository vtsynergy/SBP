/***
 * Stores a Graph.
 */
#ifndef SBP_EVALUATE_HPP
#define SBP_EVALUATE_HPP

#include <set>

#include "hungarian.hpp"

#include "graph.hpp"
#include "blockmodel/blockmodel.hpp"

namespace evaluate {

double calculate_f1_score(const Graph &graph, Hungarian::Matrix &contingency_table);

double evaluate_blockmodel(const Graph &graph, Blockmodel &blockmodel);

Hungarian::Matrix hungarian(const Graph &graph, Blockmodel &blockmodel);

}

#endif // SBP_EVALUATE_HPP
