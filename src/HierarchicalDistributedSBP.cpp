
#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nans or infs
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

#include "args.hpp"
#include "block_merge.hpp"
#include "blockmodel/blockmodel.hpp"
#include "distributed/dist_sbp.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
#include "rng.hpp"
#include "sample.hpp"
#include "sbp.hpp"


MPI_t mpi;
Args args;

double sample_time = 0.0;
double sample_extend_time = 0.0;
double finetune_time = 0.0;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime) {
    std::vector<sbp::intermediate> intermediate_results;
    if (mpi.num_processes > 1) {
        intermediate_results = sbp::dist::get_intermediates();
    } else {
        intermediate_results = sbp::get_intermediates();
    }
    std::ostringstream filepath_stream;
    filepath_stream << args.csv << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.type << ".csv";
    std::string filepath = filepath_stream.str();
    bool file_exists = fs::exists(filepath);
    std::cout << std::boolalpha <<  "writing results to " << filepath << " exists = " << file_exists << std::endl;
    fs::create_directories(fs::path(filepath_dir));
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if (!file_exists) {
        file << "tag, numvertices, numedges, overlap, blocksizevar, undirected, algorithm, iteration, mdl, "
             << "normalized_mdl_v1, sample_size, modularity, f1_score, nmi, true_mdl, true_mdl_v1, sampling_algorithm, "
             << "runtime, sampling_time, sample_extend_time, finetune_time, mcmc_iterations, mcmc_time, "
             << "sequential_mcmc_time, parallel_mcmc_time, vertex_move_time, mcmc_moves, block_merge_time, "
             << "block_merge_loop_time, blockmodel_build_time, first_blockmodel_build_time, sort_time, "
             << "load_balancing_time, access_time, update_assignmnet, total_time" << std::endl;
    }
    for (const sbp::intermediate &temp : intermediate_results) {
        file << args.tag << ", " << graph.num_vertices() << ", " << graph.num_edges() << ", " << args.overlap << ", "
             << args.blocksizevar << ", " << args.undirected << ", " << args.algorithm << ", " << temp.iteration << ", "
             << temp.mdl << ", " << temp.normalized_mdl_v1 << ", " << args.samplesize << ", "
             << temp.modularity << ", " << eval.f1_score << ", " << eval.nmi << ", " << eval.true_mdl << ", "
             << entropy::normalize_mdl_v1(eval.true_mdl, graph.num_edges()) << ", "
             << args.samplingalg << ", " << runtime << ", " << sample_time << ", " << sample_extend_time << ", "
             << finetune_time << ", " << temp.mcmc_iterations << ", " << temp.mcmc_time << ", "
             << temp.mcmc_sequential_time << ", " << temp.mcmc_parallel_time << ", "
             << temp.mcmc_vertex_move_time << ", " << temp.mcmc_moves << ", " << temp.block_merge_time << ", "
             << temp.block_merge_loop_time << ", " << temp.blockmodel_build_time << ", "
             << temp.blockmodel_first_build_time << ", " << temp.sort_time << ", " << temp.load_balancing_time << ", "
             << temp.access_time << ", " << temp.update_assignment << ", " << temp.total_time << std::endl;
    }
    file.close();
}

void evaluate_partition(Graph &graph, Blockmodel &blockmodel, double runtime) {
    if (mpi.rank != 0) return;
    evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
    std::cout << "Final F1 score = " << result.f1_score << std::endl;
    std::cout << "Community detection runtime = " << runtime << "s" << std::endl;
    write_results(graph, result, runtime);
}

void run(Partition &partition) {
    if (mpi.num_processes > 1) {
//        MPI_Barrier(MPI_COMM_WORLD);  // keep start - end as close as possible for all processes
//        double start = MPI_Wtime();
        partition.blockmodel = sbp::dist::stochastic_block_partition(partition.graph, args);
//        double end = MPI_Wtime();
//        if (mpi.rank == 0)
//        evaluate_partition(partition.graph, partition.blockmodel, end - start);
    } else {
//        auto start = std::chrono::steady_clock::now();
        partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args);
//        auto end = std::chrono::steady_clock::now();
//        evaluate_partition(partition.graph, partition.blockmodel, std::chrono::duration<double>(end - start).count());
    }
}

int GlobalRank;
int TotalRanks;

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // long rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &TotalRanks);
    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    // Heracles
    // HiERArChical distributed Stochastic block partitioning

    // New argument : subgraphs
    // Initially, there are `subgraphs` groups | subgraphs <= mpi.num_processes
    // Rank within subgraph = rank % subgraphs
    // Need to pass communicator to distributed SBP functions for communication purposes
    // Need to ensure that every rank within a subgraph gets the same subgraph

    args = Args(argc, argv);

    int ranks_in_color = ceil(double(TotalRanks) / double(args.subgraphs));
    int color = GlobalRank / ranks_in_color;
//    if (args.subgraphs == 1) color = 0;
    MPI_Comm_split(MPI_COMM_WORLD, color, GlobalRank % args.subgraphs, &mpi.comm);
    rng::init_generators();

    MPI_Comm_rank(mpi.comm, &mpi.rank);
    MPI_Comm_size(mpi.comm, &mpi.num_processes);

    std::cout << "G" << GlobalRank << " L" << mpi.rank << " (" << color << ") | can see " << mpi.num_processes << " processes." << std::endl;

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << TotalRanks << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load();
    sample::Sample subgraph = sample::round_robin(graph, color, args.subgraphs);

    std::cout << "G" << GlobalRank << " L" << mpi.rank << " (" << color << ") | can see " << subgraph.graph.num_vertices() << " V and E = " << subgraph.graph.num_edges() << std::endl;

    std::cout << "G" << GlobalRank << " L" << mpi.rank << " (" << color << ") | can see " << mpi.num_processes << " processes and is processing G with size: " << subgraph.graph.num_vertices() << std::endl;

    Partition partition;
    double start = MPI_Wtime();
    partition.graph = std::move(subgraph.graph);
    // TODO: add stopping at golden ratio
//    partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args, true);
    partition.blockmodel = sbp::dist::stochastic_block_partition(partition.graph, args, true);
    double end_blockmodeling = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << mpi.rank << " took " << end_blockmodeling - start << "s to finish initial partitioning | final B = "
              << partition.blockmodel.getNum_blocks() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    // Checking tree structure
    int partner;
    bool done = false;
    unsigned bitmask = 1;
    MPI_Status status;
    int level = 0;
    int NUM_VERTICES_TAG = 0;
    int VERTICES_TAG = 1;
    int BLOCKS_TAG = 2;

    if (mpi.rank > 0) {  // Only rank 0 from each subgraph should be involved in the merge step
        done = true;
    }

    while (!done && bitmask < args.subgraphs) {  // If subgraphs == 1, this whole thing is skipped
        partner = ranks_in_color * (color ^ bitmask);
        std::cout << "Level " << level << " | rank " << GlobalRank << "'s partner = " << partner << std::endl;
        if (partner >= TotalRanks) {
            bitmask <<=1;
            continue;
        } else if (GlobalRank < partner) {
            // The receiver
            int partner_num_vertices;
            MPI_Recv(&partner_num_vertices, 1, MPI_INT, partner, NUM_VERTICES_TAG, MPI_COMM_WORLD, &status);
            std::cout << "Level " << level << " | rank " << GlobalRank << "'s partner has " << partner_num_vertices << "vertices" << std::endl;
            std::vector<long> partner_vertices = utils::constant<long>(partner_num_vertices, -1);
            std::vector<long> partner_assignment = utils::constant<long>(partner_num_vertices, -1);
            MPI_Recv(partner_vertices.data(), partner_num_vertices, MPI_LONG, partner, VERTICES_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(partner_assignment.data(), partner_num_vertices, MPI_LONG, partner, BLOCKS_TAG, MPI_COMM_WORLD, &status);
            std::cout << "Level " << level << " | rank " << GlobalRank << " received info from pato rtner" << std::endl;
            // Build combined graph
            long index = subgraph.graph.num_vertices();
            std::vector<long> combined_mapping = subgraph.mapping;
            for (long vertex : partner_vertices) {
                combined_mapping[vertex] = index;
                index++;
            }
            std::vector<long> combined_vertices = partner_vertices;
            std::cout << GlobalRank << " | combined_vertices starting size = " << combined_vertices.size() << std::endl;
            for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
                if (subgraph.mapping[vertex] >= 0) {
                    combined_vertices.push_back(vertex);
                }
            }
            subgraph = sample::from_vertices(graph, combined_vertices, combined_mapping);
            std::cout << GlobalRank << " build combined subgraph with V = " << subgraph.graph.num_vertices() << std::endl;
            // Build combined blockmodel
            long my_num_blocks = partition.blockmodel.getNum_blocks();
            long combined_num_blocks = partition.blockmodel.getNum_blocks();
            std::vector<long> combined_block_assignment = utils::constant<long>(subgraph.graph.num_vertices(), -1);
            for (long index = 0; index < partition.blockmodel.block_assignment().size(); ++index) {
                long assignment = partition.blockmodel.block_assignment(index);
                combined_block_assignment[index] = assignment;
            }
            for (long index = 0; index < partner_num_vertices; ++index) {
                long partner_vertex = partner_vertices[index];
                long partner_block = partner_assignment[index];
                long combined_vertex = combined_mapping[partner_vertex];
                long combined_block = partner_block + partition.blockmodel.getNum_blocks();
                combined_num_blocks = std::max(combined_block + 1, combined_num_blocks);
                combined_block_assignment[combined_vertex] = combined_block;
            }
            std::cout << GlobalRank << " | combined assignment with max = "
                      << *(std::max_element(combined_block_assignment.begin(), combined_block_assignment.end()))
                      << " and B = " << combined_num_blocks << std::endl;
            partition.blockmodel = Blockmodel(combined_num_blocks, subgraph.graph, 0.5, combined_block_assignment);
            std::cout << GlobalRank << " build combined blockmodel with B = " << partition.blockmodel.getNum_blocks() << std::endl;
            // Merge blockmodels
            long partner_num_blocks = combined_num_blocks - my_num_blocks;
            std::vector<long> merge_from_blocks, merge_to_blocks;
            MapVector<std::pair<long, double>> best_merges;
            if (my_num_blocks < partner_num_blocks) {
                merge_from_blocks = utils::range<long>(0, my_num_blocks);
                merge_to_blocks = utils::range<long>(my_num_blocks, partner_num_blocks);
            } else {
                merge_from_blocks = utils::range<long>(my_num_blocks, partner_num_blocks);
                merge_to_blocks = utils::range<long>(0, my_num_blocks);
            }
            std::vector<long> block_map = utils::range<long>(0, partition.blockmodel.getNum_blocks());
            for (long merge_from : merge_from_blocks) {
                best_merges[merge_from] = std::make_pair<long, double>(-1, std::numeric_limits<double>::max());
                for (long merge_to : merge_to_blocks) {
                    // Calculate the delta entropy given the current block assignment
                    EdgeWeights out_blocks = partition.blockmodel.blockmatrix()->outgoing_edges(merge_from);
                    EdgeWeights in_blocks = partition.blockmodel.blockmatrix()->incoming_edges(merge_from);
                    long k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
                    long k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
                    long k = k_out + k_in;
                    utils::ProposalAndEdgeCounts proposal{merge_to, k_out, k_in, k};
                    Delta delta = block_merge::blockmodel_delta(merge_from, proposal.proposal, partition.blockmodel);
                    long proposed_block_self_edges = partition.blockmodel.blockmatrix()->get(merge_to, merge_to)
                                                    + delta.get(merge_to, merge_to);
                    double dE = entropy::block_merge_delta_mdl(merge_from, proposal, partition.blockmodel, delta);
                    if (dE < best_merges[merge_from].second) {
                        best_merges[merge_from] = std::make_pair(merge_to, dE);
                        block_map[merge_from] = merge_to;
                    }
                }
            }
            std::vector<long> assignment = partition.blockmodel.block_assignment();
            for (long i = 0; i < subgraph.graph.num_vertices(); ++i) {
                assignment[i] = block_map[assignment[i]];
            }
            std::vector<long> mapping = Blockmodel::build_mapping(assignment);
            for (size_t i = 0; i < assignment.size(); ++i) {
                long block = assignment[i];
                long new_block = mapping[block];
                assignment[i] = new_block;
            }
            partition.blockmodel = Blockmodel((long) merge_to_blocks.size(), subgraph.graph, 0.5, assignment);
            std::cout << GlobalRank << " | merged Blockmodel to one with B = " << partition.blockmodel.getNum_blocks() << std::endl;
            bitmask <<= 1;
        } else {
            // The sender
            // Send number of vertices
            int local_num_vertices = subgraph.graph.num_vertices();
            std::vector<long> local_vertices = utils::constant<long>(subgraph.graph.num_vertices(), -1);
            std::vector<long> local_assignment = utils::constant<long>(subgraph.graph.num_vertices(), -1);
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(graph, subgraph, partition, local_vertices, local_assignment)
            for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
                long subgraph_index = subgraph.mapping[vertex];
                if (subgraph_index < 0) continue;  // vertex not present
                long assignment = partition.blockmodel.block_assignment(subgraph_index);
                local_vertices[subgraph_index] = vertex;
                local_assignment[subgraph_index] = assignment;
            }
            MPI_Send(&local_num_vertices, 1, MPI_INT, partner, NUM_VERTICES_TAG, MPI_COMM_WORLD);
            MPI_Send(local_vertices.data(), local_num_vertices, MPI_LONG, partner, VERTICES_TAG, MPI_COMM_WORLD);
            MPI_Send(local_assignment.data(), local_num_vertices, MPI_LONG, partner, BLOCKS_TAG, MPI_COMM_WORLD);
            done = true;
        }
        level++;
    }

    if (GlobalRank == 0) {
        std::cout << "==================================== FINAL FINETUNING ============================" << std::endl;
        std::cout << "Graph V = " << subgraph.graph.num_vertices() << " E = " << subgraph.graph.num_edges() << std::endl;
        std::cout << "Blockmodel with B = " << partition.blockmodel.getNum_blocks() << std::endl;
        // Finetune final assignment
        Blockmodel blockmodel = std::move(partition.blockmodel);
        blockmodel.setOverall_entropy(entropy::mdl(blockmodel, subgraph.graph.num_vertices(), subgraph.graph.num_edges()));
        BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        double iteration = sbp::get_intermediates().size();
        std::cout << "Performing final fine-tuning on just one rank" << std::endl;
        while (!sbp::done_blockmodeling(blockmodel, blockmodel_triplet)) {
            if (blockmodel.getNum_blocks_to_merge() != 0) {
                std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                          << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
            }
            double start_bm = MPI_Wtime();
            std::cout << "Doing the merge_blocks step yo!" << std::endl;
            blockmodel = block_merge::merge_blocks(blockmodel, subgraph.graph, subgraph.graph.num_edges());
            block_merge::BlockMerge_time += MPI_Wtime() - start_bm;
            std::cout << "Starting MCMC vertex moves" << std::endl;
            double start_mcmc = MPI_Wtime();
//        if (args.algorithm == "async_gibbs_old" && iteration < double(args.asynciterations))
//            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
//        else
            common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
            if (args.algorithm == "async_gibbs" && iteration < double(args.asynciterations))
                blockmodel = finetune::asynchronous_gibbs(blockmodel, subgraph.graph, blockmodel_triplet);
            else if (args.algorithm == "hybrid_mcmc")
                blockmodel = finetune::hybrid_mcmc(blockmodel, subgraph.graph, blockmodel_triplet);
            else // args.algorithm == "metropolis_hastings"
                blockmodel = finetune::metropolis_hastings(blockmodel, subgraph.graph, blockmodel_triplet);
            finetune::MCMC_time += MPI_Wtime() - start_mcmc;
            sbp::total_time += MPI_Wtime() - start_bm;
            sbp::add_intermediate(++iteration, subgraph.graph, -1, blockmodel.getOverall_entropy());
            blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
            common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        }
        // only last iteration result will calculate expensive modularity
        double modularity = -1;
        if (args.modularity)
            modularity = graph.modularity(blockmodel.block_assignment());
        sbp::add_intermediate(-1, subgraph.graph, modularity, blockmodel.getOverall_entropy());
        // Evaluate finetuned assignment
        // Reorder truth labels to match the reordered graph vertices
        std::vector<long> reordered_truth = utils::constant<long>(graph.num_vertices(), -1);
        for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            long mapped_index = subgraph.mapping[vertex];
            long truth_community = graph.assignment(vertex);
            reordered_truth[mapped_index] = truth_community;
        }
        double end = MPI_Wtime();
        subgraph.graph.assignment(reordered_truth);
        evaluate_partition(subgraph.graph, blockmodel, end - start);
    }
    /*
    sample::Sample detached;
    Partition partition;
    double start = MPI_Wtime();
    if (args.detach) {
        detached = sample::detach(graph);
        partition.graph = std::move(detached.graph);
        std::cout << "detached num vertices: " << partition.graph.num_vertices() << " E: "
                  << partition.graph.num_edges() << std::endl;
    } else {
        partition.graph = std::move(graph);
    }
    if (args.samplesize <= 0.0) {
        std::cerr << "ERROR " << "Sample size of " << args.samplesize << " is too low. Must be greater than 0.0" << std::endl;
        exit(-5);
    } else if (args.samplesize < 1.0) {
        double sample_start_t = MPI_Wtime();
        std::cout << "Running sampling with size: " << args.samplesize << std::endl;
//        sample::Sample s = sample::max_degree(partition.graph);
        sample::Sample s = sample::sample(partition.graph);
        Partition sample_partition;
        sample_partition.graph = std::move(s.graph);  // s.graph may be empty now
        // add timer
        double sample_end_t = MPI_Wtime();
        sample_time = sample_end_t - sample_start_t;
        run(sample_partition);
        double extend_start_t = MPI_Wtime();
        s.graph = std::move(sample_partition.graph);  // refill s.graph
        // extend sample to full graph
        partition.blockmodel = sample::extend(partition.graph, sample_partition.blockmodel, s);
        // fine-tune full graph
        double finetune_start_t = MPI_Wtime();
        partition.blockmodel = finetune::finetune_assignment(partition.blockmodel, partition.graph);
        double finetune_end_t = MPI_Wtime();
        sample_extend_time = finetune_start_t - extend_start_t;
        finetune_time = finetune_end_t - finetune_start_t;
    } else {
        std::cout << "Running without sampling." << std::endl;
        run(partition);
    }
    if (args.detach) {
        std::cout << "Reattaching island and 1-degree vertices" << std::endl;
        partition.blockmodel = sample::reattach(graph, partition.blockmodel, detached);
    } else {
        graph = std::move(partition.graph);
    }
    // evaluate
    double end = MPI_Wtime();
    evaluate_partition(graph, partition.blockmodel, end - start);
    */
    MPI_Finalize();
}
