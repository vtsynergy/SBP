#include "distributed/dist_sbp.hpp"

#include "distributed/dist_block_merge.hpp"
#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/dist_finetune.hpp"
#include "finetune.hpp"
#include "distributed/two_hop_blockmodel.hpp"

#include <sstream>

namespace sbp::dist {

void record_runtime_imbalance() {
    int recvcount = (int) finetune::dist::MCMC_RUNTIMES.size();
    std::cout << mpi.rank << " : recvcount = " << recvcount << " np = " << mpi.num_processes << std::endl;
    std::cout << mpi.rank << " : runtime[5] = " << finetune::dist::MCMC_RUNTIMES[5] << std::endl;
    std::cout << mpi.rank << " : runtimes size = " << finetune::dist::MCMC_RUNTIMES.size() << std::endl;
//    std::vector<double> all_mcmc_runtimes = utils::constant<double>(recvcount, 0);
    std::vector<double> all_mcmc_runtimes(recvcount * mpi.num_processes, 0.0);
    std::cout << mpi.rank << " : allocated vector size = " << all_mcmc_runtimes.size() << std::endl;
    MPI_Gather(finetune::dist::MCMC_RUNTIMES.data(), recvcount, MPI_DOUBLE,
               all_mcmc_runtimes.data(), recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpi.rank != 0) return;  // Only rank 0 should actually save a CSV file
    std::ostringstream filepath_stream;
    filepath_stream << args.csv << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    std::ostringstream filename_stream;
    filename_stream << args.csv << args.numvertices << "/" << args.type << "_imbalance.csv";
    std::string filepath = filename_stream.str();
    int attempt = 0;
    while (fs::exists(filepath)) {
        filename_stream = std::ostringstream();
        filename_stream << args.csv << args.numvertices << "/" << args.type << "_imbalance_" << attempt << ".csv";
        filepath = filename_stream.str();
        attempt++;
    }
    std::cout << std::boolalpha <<  "writing imbalance #s to " << filepath << std::endl;
    fs::create_directories(fs::path(filepath_dir));
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    file << "iteration, ";
    for (int i = 0; i < mpi.num_processes; ++i) {
        file << i;
        if (i == mpi.num_processes - 1) {
            file << std::endl;
        } else {
            file << ", ";
        }
    }
    for (int iteration = 0; iteration < finetune::dist::MCMC_RUNTIMES.size(); ++iteration) {
        file << iteration << ", ";
        for (int rank = 0; rank < mpi.num_processes; ++rank) {
            size_t position = rank * finetune::dist::MCMC_RUNTIMES.size() + iteration;
            file << all_mcmc_runtimes[position];
            if (rank < mpi.num_processes - 1) file << ", ";
        }
        file << std::endl;
    }
    file.close();
}

// Blockmodel stochastic_block_partition(Graph &graph, MPI &mpi, Args &args) {
Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    // DistBlockmodel blockmodel(graph, args, mpi);
    TwoHopBlockmodel blockmodel(graph.num_vertices(), graph, BLOCK_REDUCTION_RATE);
    common::candidates = std::uniform_int_distribution<int>(0, blockmodel.getNum_blocks() - 2);
    // Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    if (mpi.rank == 0)
        std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
                  << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    DistBlockmodelTriplet blockmodel_triplet = DistBlockmodelTriplet();
    int iteration = 0;
    while (!dist::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (mpi.rank == 0 && blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::dist::merge_blocks(blockmodel, graph);
        common::candidates = std::uniform_int_distribution<int>(0, blockmodel.getNum_blocks() - 2);
        if (mpi.rank == 0) std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs" && iteration < args.asynciterations)
            blockmodel = finetune::dist::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else if (args.algorithm == "hybrid_mcmc" && iteration < args.asynciterations)
            blockmodel = finetune::dist::hybrid_mcmc(blockmodel, graph, blockmodel_triplet);
        else
            blockmodel = finetune::dist::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        iteration++;
    }
    std::cout << "Total MCMC iterations: " << finetune::MCMC_iterations << std::endl;
    double modularity = -1;
    if (args.modularity)
        modularity = graph.modularity(blockmodel.block_assignment());
    add_intermediate(-1, graph, modularity, blockmodel.getOverall_entropy());
    record_runtime_imbalance();
    return blockmodel;
}

bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet, int min_num_blocks) {
    if (mpi.rank == 0) std::cout << "distributed done_blockmodeling" << std::endl;
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || !blockmodel_triplet.get(2).empty) {
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

}  // namespace sbp::dist