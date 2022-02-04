/**
 * Functions relating to calculating blockmodel entropy.
 */

#include "blockmodel/blockmodel.hpp"

namespace entropy {

/// Calculates the minimum description length of `blockmodel` for a directed graph with `num_vertices` vertices and
/// `num_edges` edges.
double mdl(const Blockmodel &blockmodel, int num_vertices, int num_edges);

// TODO: add an undirected mdl

}