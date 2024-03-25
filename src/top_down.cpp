//
// The town-down alternative approach to stochastic block blockmodeling.
// Created by wanye on 5/11/2022.
//
#include "top_down.hpp"

#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <random>
#include <vector>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "common.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "sbp.hpp"
#include "rng.hpp"
#include "utils.hpp"

namespace top_down {

void apply_best_splits(Blockmodel &blockmodel, const std::vector<Split> &best_splits,
                       const std::vector<double> &split_entropy, int target_num_communities) {
    // Sort entropies in ascending order
    std::vector<long> sorted_splits = utils::argsort<double>(split_entropy);
    // Modify assignment, increasing blockmodel.blockNum() until reaching target
    long num_blocks = blockmodel.getNum_blocks();
    long counter = 0;
    while (num_blocks < target_num_communities) {
        long block = sorted_splits[counter];
        const Split& split = best_splits[block];
        MapVector<long> reverse_translator;
        for (const auto &entry : split.translator) {
            reverse_translator[entry.second] = entry.first;
        }
        for (long index = 0; index < split.num_vertices; ++index) {
            long assignment = split.blockmodel->block_assignment(index);
            if (assignment == 1) {
                long vertex = reverse_translator[index];
                blockmodel.set_block_assignment(vertex, num_blocks);
            }
        }
        num_blocks++;
    }
    // Re-form blockmodel
    blockmodel.setNum_blocks(num_blocks);
}

Split propose_split(long community, const Graph &graph, const Blockmodel &blockmodel) {
    Split split;
    std::vector<bool> community_flag = utils::constant<bool>(graph.num_vertices(), false);
    std::vector<long> community_vertices;
    long index = 0;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (blockmodel.block_assignment(vertex) != community) continue;
        community_flag[vertex] = true;
        community_vertices.push_back(vertex);
        split.translator[vertex] = index;
        index++;
    }
    split.num_vertices = long(community_vertices.size());
    Graph subgraph(split.num_vertices);
    for (long vertex : community_vertices) {
        for (long neighbor : graph.out_neighbors(vertex)) {
            if (!community_flag[neighbor]) continue;
            subgraph.add_edge(split.translator[vertex], split.translator[neighbor]);
        }
    }
    // TODO: -1s in assignment may screw with blockmodel formation
    std::vector<long> split_assignment = utils::constant<long>(split.num_vertices, -1);
    std::uniform_int_distribution<long> distribution(0, 1);
    // TODO: implement translator for vertex IDs, and store it in Split
    for (long vertex = 0; vertex < split.num_vertices; ++vertex) {
        split_assignment[vertex] = distribution(rng::generator());
    }
    split.blockmodel = std::make_shared<Blockmodel>(2, subgraph, 0.5, split_assignment);
    split.num_edges = subgraph.num_edges();
    split.subgraph = subgraph;
    return split;
}

Blockmodel run(const Graph &graph) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph, float(BLOCK_REDUCTION_RATE));
    double initial_mdl = entropy::nonparametric::mdl(blockmodel, graph);
//    double initial_mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    sbp::add_intermediate(0, graph, -1, initial_mdl);
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    float iteration = 0;
    while (!sbp::done_blockmodeling(blockmodel, blockmodel_triplet)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = split_communities(blockmodel, graph, blockmodel.getNum_blocks() * 2);
        if (iteration < 1) {
            double mdl = entropy::nonparametric::mdl(blockmodel, graph);  // .num_vertices(), graph.num_edges());
            sbp::add_intermediate(0.5, graph, -1, mdl);
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start = MPI_Wtime();
        if (args.algorithm == "async_gibbs" && iteration < float(args.asynciterations))
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        else if (args.algorithm == "hybrid_mcmc")
            blockmodel = finetune::hybrid_mcmc(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        else // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
//        iteration++;
        finetune::MCMC_time += MPI_Wtime() - start;
        sbp::add_intermediate(++iteration, graph, -1, blockmodel.getOverall_entropy());
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    }
    return blockmodel;
}

Blockmodel split_communities(Blockmodel &blockmodel, const Graph &graph, int target_num_communities) {
    int num_blocks = blockmodel.getNum_blocks();
    std::vector<Split> best_split_for_each_block(num_blocks);
    std::vector<double> delta_entropy_for_each_block =
            utils::constant<double>(num_blocks, std::numeric_limits<double>::max());
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            Split split = propose_split(current_block, graph, blockmodel);
            // TODO: currently computing delta entropy for the split ONLY. Can we compute dE for entire blockmodel?
            double new_entropy = entropy::nonparametric::mdl(*(split.blockmodel), split.subgraph);  // split.num_vertices, split.num_edges);
            double old_entropy = entropy::null_mdl_v1(split.subgraph);
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

