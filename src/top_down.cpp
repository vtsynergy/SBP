//
// The town-down alternative approach to stochastic block blockmodeling.
// Created by wanye on 5/11/2022.
//
#include "top_down.hpp"

#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "blockmodel/blockmodel.hpp"
#include "common.hpp"
#include "entropy.hpp"
#include "graph.hpp"
#include "utils.hpp"

namespace top_down {

void apply_best_splits(Blockmodel &blockmodel, const std::vector<Split> &best_splits,
                       const std::vector<double> &split_entropy, int target_num_communities) {
    // Sort entropies in ascending order
    std::vector<int> sorted_splits = utils::sort_indices(split_entropy);
    // Modify assignment, increasing blockmodel.blockNum() until reaching target
    int num_blocks = blockmodel.getNum_blocks();
    int counter = 0;
    while (num_blocks < target_num_communities) {
        int block = sorted_splits[counter];
        const Split& split = best_splits[block];
        MapVector<int> reverse_translator;
        for (const auto &entry : split.translator) {
            reverse_translator[entry.second] = entry.first;
        }
        for (int index = 0; index < split.num_vertices; ++index) {
            int assignment = split.blockmodel->block_assignment(index);
            if (assignment == 1) {
                int vertex = reverse_translator[index];
                blockmodel.set_block_assignment(vertex, num_blocks);
            }
        }
        num_blocks++;
    }
    // Re-form blockmodel
    blockmodel.setNum_blocks(num_blocks);
}

Split propose_split(int community, const Graph &graph, const Blockmodel &blockmodel) {
    Split split;
    std::vector<bool> community_flag = utils::constant<bool>(graph.num_vertices(), false);
    std::vector<int> community_vertices;
    int index = 0;
    for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (blockmodel.block_assignment(vertex) != community) continue;
        community_flag[vertex] = true;
        community_vertices.push_back(vertex);
        split.translator[vertex] = index;
        index++;
    }
    split.num_vertices = int(community_vertices.size());
    Graph subgraph(split.num_vertices);
    for (int vertex : community_vertices) {
        for (int neighbor : graph.out_neighbors(vertex)) {
            if (!community_flag[neighbor]) continue;
            subgraph.add_edge(split.translator[vertex], split.translator[neighbor]);
        }
    }
    // TODO: -1s in assignment may screw with blockmodel formation
    std::vector<int> split_assignment = utils::constant<int>(split.num_vertices, -1);
    std::uniform_int_distribution<int> distribution(0, 1);
    // TODO: implement translator for vertex IDs, and store it in Split
    for (int vertex = 0; vertex < split.num_vertices; ++vertex) {
        split_assignment[vertex] = distribution(common::generator);
    }
    split.blockmodel = std::make_shared<Blockmodel>(2, subgraph, 0.5, split_assignment);
    split.num_edges = subgraph.num_edges();
    return split;
}

Blockmodel split_communities(Blockmodel &blockmodel, const Graph &graph, int target_num_communities) {
    int num_blocks = blockmodel.getNum_blocks();
    std::vector<Split> best_split_for_each_block(num_blocks);
    std::vector<double> delta_entropy_for_each_block =
            utils::constant<double>(num_blocks, std::numeric_limits<double>::max());
//    std::vector<int> block_assignment = utils::range<int>(0, num_blocks);
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            Split split = propose_split(current_block, graph, blockmodel);
            // TODO: currently computing delta entropy for the split ONLY. Can we compute dE for entire blockmodel?
            double new_entropy = entropy::mdl(*(split.blockmodel), split.num_vertices, split.num_edges);
            double old_entropy = entropy::null_mdl_v1(split.num_edges);
            double delta_entropy = new_entropy - old_entropy;
            if (delta_entropy < delta_entropy_for_each_block[current_block]) {
                delta_entropy_for_each_block[current_block] = delta_entropy;
                best_split_for_each_block[current_block] = split;
            }
        }
    }
    apply_best_splits(blockmodel, best_split_for_each_block, delta_entropy_for_each_block, target_num_communities);
    blockmodel.initialize_edge_counts(graph);
    return blockmodel;
}

}

