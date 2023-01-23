//
// Created by Frank on 1/6/2023.
//

#include "agglomerative.hpp"

#include "entropy.hpp"
#include "finetune.hpp"
#include "block_merge.hpp"

#include <omp.h>
#include <set>

namespace agglomerative {

double total_time = 0.0;
double Blockmodel_first_build_time = 0.0;

std::vector<Intermediate> intermediate_results;
std::vector<Intermediate> get_intermediates() {
    return intermediate_results;
}

void add_intermediate(float iteration, const Graph &graph, double modularity, double mdl) {
    double normalized_mdl_v1 = entropy::normalize_mdl_v1(mdl, graph.num_edges());
    Intermediate intermediate {};
    intermediate.iteration = iteration;
    intermediate.mdl = mdl;
    intermediate.normalized_mdl_v1 = normalized_mdl_v1;
    intermediate.modularity = modularity;
    intermediate.mcmc_iterations = finetune::MCMC_iterations;
    intermediate.mcmc_time = finetune::MCMC_time;
    intermediate.mcmc_sequential_time = finetune::MCMC_sequential_time;
    intermediate.mcmc_parallel_time = finetune::MCMC_parallel_time;
    intermediate.mcmc_vertex_move_time = finetune::MCMC_vertex_move_time;
    intermediate.mcmc_moves = finetune::MCMC_moves;
    intermediate.block_merge_time = block_merge::BlockMerge_time;
    intermediate.block_merge_loop_time = block_merge::BlockMerge_loop_time;
    intermediate.blockmodel_build_time = BLOCKMODEL_BUILD_TIME;
    intermediate.blockmodel_first_build_time = Blockmodel_first_build_time;
    intermediate.sort_time = Blockmodel_sort_time;
    intermediate.access_time = Blockmodel_access_time;
    intermediate.total_time = total_time;
    intermediate.update_assignment = Blockmodel_update_assignment;
    intermediate_results.push_back(intermediate);
    std::cout << "Iteration " << iteration << " MDL: " << mdl << " v1 normalized: " << normalized_mdl_v1
              << " modularity: " << modularity << " MCMC iterations: " << finetune::MCMC_iterations << " MCMC time: "
              << finetune::MCMC_time << " Block Merge time: " << block_merge::BlockMerge_time << " total time: "
              << total_time << std::endl;
}

Blockmodel agglomerate(Blockmodel &blockmodel, const Graph &graph) {

}

Blockmodel move_vertices(Blockmodel &blockmodel, const Graph &graph) {
    Blockmodel supernode_blockmodel = blockmodel.copy();
    std::vector<int> block_map = utils::range<int>(0, blockmodel.getNum_blocks());
    auto translate = [&block_map](int block) -> int {
        int b = block;
        do {
            if (b >= block_map.size())
                std::cout << "bm[" << b << "] = " << block_map[b] << std::endl;
            b = block_map[b];
        } while (block_map[b] != b);
        if (b >= block_map.size())
            std::cout << "final bm[" << block << "] = " << block_map[b] << std::endl;
        return b;
    };
    int total_vertex_moves = 0;
    std::vector<double> delta_entropies;
    for (int iteration = 0; iteration < finetune::MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        double start_t = MPI_Wtime();
        for (int vertex = 0; vertex < supernode_graph.num_vertices(); ++vertex) {
            // TODO: implement propose_move
            // NOTE: some moves will decrease the number of blocks, while other's won't
            VertexMove proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        finetune::MCMC_sequential_time += MPI_Wtime() - start_t;
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        finetune::MCMC_iterations++;
        // Early stopping
        // TODO: implement early stopping
        if (early_stop(iteration, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    finetune::MCMC_moves += total_vertex_moves;
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

Blockmodel run(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads)
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph, float(BLOCK_REDUCTION_RATE));
    Blockmodel_first_build_time = BLOCKMODEL_BUILD_TIME;
    BLOCKMODEL_BUILD_TIME = 0.0;
    double initial_mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    add_intermediate(0, graph, -1, initial_mdl);
    float iteration = 0;
    int initial_num_blocks, current_num_blocks;
    Graph supernode_graph = graph;
    do {
        std::cout << "Starting vertex moves" << std::endl;
        double start_iteration = MPI_Wtime();
        initial_num_blocks = blockmodel.getNum_blocks();
        blockmodel = move_vertices(blockmodel, supernode_graph);
        std::cout << "Starting block agglomeration" << std::endl;
        blockmodel = agglomerate(blockmodel, supernode_graph);
        current_num_blocks = blockmodel.getNum_blocks();
        total_time += MPI_Wtime() - start_iteration;
        add_intermediate(++iteration, graph, -1, blockmodel.getOverall_entropy());
        // TODO: implement blockmodel to graph conversion
        supernode_graph = blockmodel_to_graph(blockmodel);
    } while (current_num_blocks != initial_num_blocks);
    // TODO: make sure this runs metropolis hastings on the whole graph
    blockmodel = move_vertices(blockmodel, graph);
    double modularity = -1;
    if (args.modularity)
        modularity = graph.modularity(blockmodel.block_assignment());
    add_intermediate(-1, graph, modularity, blockmodel.getOverall_entropy());
    return blockmodel;
}

} // namespace agglomerative
