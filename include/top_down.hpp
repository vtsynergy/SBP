/**
 * The town-down alternative approach to stochastic block blockmodeling.
 */
#ifndef SBP_TOP_DOWN_HPP
#define SBP_TOP_DOWN_HPP

#include <memory>
#include <vector>

#include "blockmodel/blockmodel.hpp"
#include "graph.hpp"

namespace top_down {

/// Stores information about a community split.
struct Split {
    /// The blockmodel for the two split communities.
    std::shared_ptr<Blockmodel> blockmodel;
    /// The number of vertices involved in the split.
    long num_vertices;
    /// The number of edges involved in the split.
    long num_edges;
    /// Translates full graph vertex IDs to subgraph vertex IDs
    MapVector<long> translator;
    /// The subgraph containing the vertices in the original block
    Graph subgraph;
};

static const long NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

/// Accepts or rejects a blockmodel split.
bool accept(const Split &split, const Blockmodel &blockmodel);

/// Applies the best community splits.
void apply_best_splits(const Blockmodel &blockmodel, const std::vector<Split> &best_splits,
                       const std::vector<double> &split_entropy, int target_num_communities);

/// Applies the community split to `blockmodel`.
void apply_split(const Split &split, Blockmodel &blockmodel);

/// Splits a single community into two. Returns a blockmodel containing just 2 communities that resulted from
/// splitting community `community`.
Split propose_split(long community, const Graph &graph, const Blockmodel &blockmodel);

/// Runs the top-down community detection algorithm.
Blockmodel run(const Graph &graph);

/// The reverse of block_merge::merge_blocks. Proposes several community splits, and applies the best ones until the
/// number of communities reaches `target_num_communities`.
Blockmodel split_communities(Blockmodel &blockmodel, const Graph &graph, int target_num_communities);

}

#endif // SBP_TOP_DOWN_HPP