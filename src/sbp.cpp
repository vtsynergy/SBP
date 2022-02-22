#include "sbp.hpp"

#include "block_merge.hpp"
#include "blockmodel/dist_blockmodel.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "fs.hpp"
#include "mpi_data.hpp"

#include "assert.h"
#include <sstream>

namespace sbp {

std::vector<Intermediate> intermediate_results;

std::vector<Intermediate> get_intermediates() {
    return intermediate_results;
}

void write_results(float iteration, std::ofstream &file, const Graph &graph, const Blockmodel &blockmodel, double mdl) {
    file << args.tag << "," << graph.num_vertices() << "," << args.overlap << "," << args.blocksizevar << ",";
    file << args.undirected << "," << args.algorithm << "," << iteration << ",";
    file << mdl << ","  << entropy::normalize_mdl_v1(mdl, graph.num_edges()) << ",";
    file << entropy::normalize_mdl_v2(mdl, graph.num_vertices(), graph.num_edges()) << ",";
    file << graph.modularity(blockmodel.block_assignment()) << "," << blockmodel.interblock_edges() << ",";
    file << blockmodel.block_size_variation() << std::endl;
}

void add_intermediate(float iteration, const Graph &graph, const Blockmodel &blockmodel, double mdl) {
    double normalized_mdl_v1 = entropy::normalize_mdl_v1(mdl, graph.num_edges());
    double normalized_mdl_v2 = entropy::normalize_mdl_v2(mdl, graph.num_vertices(), graph.num_edges());
    double modularity = graph.modularity(blockmodel.block_assignment());
    double interblock_edges = blockmodel.interblock_edges();
    double block_size_variation = blockmodel.block_size_variation();
    intermediate_results.push_back(Intermediate { iteration, mdl, normalized_mdl_v1, normalized_mdl_v2,
            modularity, interblock_edges, block_size_variation });
    std::cout << "Iteration " << iteration << " MDL: " << mdl << " v1 normalized: " << normalized_mdl_v1
              << " v2 normalized: " << normalized_mdl_v2 << " modularity: " << modularity
              << " interblock edge %: " << interblock_edges << " block size variation: " << block_size_variation
              << std::endl;
}

Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), float(BLOCK_REDUCTION_RATE));
    double initial_mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
//    double initial_modularity = graph.modularity(blockmodel.block_assignment());
//    double null_model_mdl_v1 = entropy::null_mdl_v1(graph.num_edges());
//    double null_model_mdl_v2 = entropy::null_mdl_v2(graph.num_vertices(), graph.num_edges());
//    std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
//              << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
//    std::cout << "Initial MDL = " << initial_mdl
//              << " log posterior probability = " << blockmodel.log_posterior_probability(graph.num_edges())
//              << " Modularity = " << initial_modularity << std::endl;
    add_intermediate(0, graph, blockmodel, initial_mdl);
//    write_results(0, file, graph, blockmodel, initial_mdl);
//    std::cout << "Null model MDL v1 = " << null_model_mdl_v1 << " v2 = " << null_model_mdl_v2 << std::endl;
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    float iteration = 0;
    while (!done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::merge_blocks(blockmodel, graph.out_neighbors(), graph.num_edges());
        if (iteration < 1) {
            double mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
            add_intermediate(0.5, graph, blockmodel, mdl);
//            write_results(0.5, file, graph, blockmodel, mdl);
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else  // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
//        iteration++;
        add_intermediate(++iteration, graph, blockmodel, blockmodel.getOverall_entropy());
//        write_results(iteration, file, graph, blockmodel, blockmodel.getOverall_entropy());
//        std::cout << "Iteration " << iteration << ": MDL = " << blockmodel.getOverall_entropy()
//                  << " log posterior probability = " << blockmodel.log_posterior_probability(graph.num_edges())
//                  << " Modularity = " << graph.modularity(blockmodel.block_assignment()) << std::endl;
//        std::cout << "interblock E = " << blockmodel.interblock_edges() << " var = "
//                  << blockmodel.block_size_variation() << " composite = " << blockmodel.difficulty_score() << std::endl;
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    }
//    file.close();
    return blockmodel;
}

bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks) {
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || (blockmodel_triplet.get(2).empty == false)) {
            return true;
        }
    }
    if (blockmodel_triplet.optimal_num_blocks_found) {
        blockmodel_triplet.status();
        std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}

namespace dist {

// Blockmodel stochastic_block_partition(Graph &graph, MPI &mpi, Args &args) {
Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    // DistBlockmodel blockmodel(graph, args, mpi);
    TwoHopBlockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    // Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    if (mpi.rank == 0)
        std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
                  << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    DistBlockmodelTriplet blockmodel_triplet = DistBlockmodelTriplet();
    while (!dist::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (mpi.rank == 0 && blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::dist::merge_blocks(blockmodel, graph.out_neighbors(), graph.num_edges());
        if (mpi.rank == 0) std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::dist::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else
            blockmodel = finetune::dist::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    }
    std::cout << "Total MCMC iterations: " << finetune::num_iterations << std::endl;
    return blockmodel;
}

bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet, int min_num_blocks) {
    if (mpi.rank == 0) std::cout << "distributed done_blockmodeling" << std::endl;
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || (blockmodel_triplet.get(2).empty == false)) {
            return true;
        }
    }
    if (blockmodel_triplet.optimal_num_blocks_found) {
        blockmodel_triplet.status();
        std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}

}  // namespace dist

}  // namespace sbp