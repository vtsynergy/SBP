/**
 * Functions relating to calculating blockmodel entropy.
 */

#include "blockmodel/blockmodel.hpp"
#include "common.hpp"
#include "utils.hpp"

namespace entropy {

/// Computes the change in blockmodel minimum description length when a vertex moves from `current_block` to `proposal`.
/// Uses a dense version of `updates` to the blockmodel, and requires pre-calculated updated `block_degrees`.
double delta_mdl(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                 EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in blockmodel minimum description length when a vertex moves from `current_block` to `proposal`.
/// Uses a sparse version of `updates` to the blockmodel, and requires pre-calculated updated `block_degrees`.
double delta_mdl(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                 SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in blockmodel minimum description length when a vertex moves from one block to another. Uses
/// changes to the blockmodel, stored in `delta`, to perform the computation, and does not require pre-calculated
/// updated block_degrees. This method should be preferred in almost all cases.
double delta_mdl(const Blockmodel &blockmodel, const Delta &delta, const utils::ProposalAndEdgeCounts &proposal);

/// Calculates the minimum description length of `blockmodel` for a directed graph with `num_vertices` vertices and
/// `num_edges` edges.
double mdl(const Blockmodel &blockmodel, int num_vertices, int num_edges);

// TODO: add an undirected mdl

}