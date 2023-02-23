
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
    std::vector<sbp::Intermediate> intermediate_results;
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
    for (const sbp::Intermediate &temp : intermediate_results) {
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

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // int rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.num_processes);
    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    args = Args(argc, argv);
    rng::init_generators();

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load();
    sample::Sample subgraph = sample::round_robin(graph);
    Partition partition;
    double start = MPI_Wtime();
    partition.graph = std::move(subgraph.graph);
    // TODO: add stopping at golden ratio
    partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args, true);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_blockmodeling = MPI_Wtime();
    std::cout << "Rank " << mpi.rank << " took " << end_blockmodeling - start << "s to finish runtime | final B = "
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
    while (!done && bitmask < mpi.num_processes) {
        partner = mpi.rank ^ bitmask;
        std::cout << "Level " << level << " | rank " << mpi.rank << "'s partner = " << partner << std::endl;
        if (partner >= mpi.num_processes) {
            bitmask <<=1;
            continue;
        } else if (mpi.rank < partner) {
            // The receiver
            int partner_num_vertices;
            MPI_Recv(&partner_num_vertices, 1, MPI_INT, partner, NUM_VERTICES_TAG, MPI_COMM_WORLD, &status);
            std::cout << "Level " << level << " | rank " << mpi.rank << "'s partner has " << partner_num_vertices << "vertices" << std::endl;
            std::vector<int> partner_vertices = utils::constant<int>(partner_num_vertices, -1);
            std::vector<int> partner_assignment = utils::constant<int>(partner_num_vertices, -1);
            MPI_Recv(partner_vertices.data(), partner_num_vertices, MPI_INT, partner, VERTICES_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(partner_assignment.data(), partner_num_vertices, MPI_INT, partner, BLOCKS_TAG, MPI_COMM_WORLD, &status);
            std::cout << "Level " << level << " | rank " << mpi.rank << " received info from partner" << std::endl;
            // Build combined graph
            int index = subgraph.graph.num_vertices();
            std::vector<int> combined_mapping = subgraph.mapping;
            for (int vertex : partner_vertices) {
                combined_mapping[vertex] = index;
                index++;
            }
            std::vector<int> combined_vertices = partner_vertices;
            std::cout << mpi.rank << " | combined_vertices starting size = " << combined_vertices.size() << std::endl;
            for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
                if (subgraph.mapping[vertex] >= 0) {
                    combined_vertices.push_back(vertex);
                }
            }
            subgraph = sample::from_vertices(graph, combined_vertices, combined_mapping);
            std::cout << mpi.rank << " build combined subgraph with V = " << subgraph.graph.num_vertices() << std::endl;
            // Build combined blockmodel
            int my_num_blocks = partition.blockmodel.getNum_blocks();
            int combined_num_blocks = partition.blockmodel.getNum_blocks();
            std::vector<int> combined_block_assignment = utils::constant<int>(subgraph.graph.num_vertices(), -1);
            for (int index = 0; index < partition.blockmodel.block_assignment().size(); ++index) {
                int assignment = partition.blockmodel.block_assignment(index);
                combined_block_assignment[index] = assignment;
            }
            for (int index = 0; index < partner_num_vertices; ++index) {
                int partner_vertex = partner_vertices[index];
                int partner_block = partner_assignment[index];
                int combined_vertex = combined_mapping[partner_vertex];
                int combined_block = partner_block + partition.blockmodel.getNum_blocks();
                combined_num_blocks = std::max(combined_block + 1, combined_num_blocks);
                combined_block_assignment[combined_vertex] = combined_block;
            }
            std::cout << mpi.rank << " | combined assignment with max = "
                      << *(std::max_element(combined_block_assignment.begin(), combined_block_assignment.end()))
                      << " and B = " << combined_num_blocks << std::endl;
            partition.blockmodel = Blockmodel(combined_num_blocks, subgraph.graph, 0.5, combined_block_assignment);
            std::cout << mpi.rank << " build combined blockmodel with B = " << partition.blockmodel.getNum_blocks() << std::endl;
            // Merge blockmodels
            int partner_num_blocks = combined_num_blocks - my_num_blocks;
            std::vector<int> merge_from_blocks, merge_to_blocks;
            MapVector<std::pair<int, double>> best_merges;
            if (my_num_blocks < partner_num_blocks) {
                merge_from_blocks = utils::range<int>(0, my_num_blocks);
                merge_to_blocks = utils::range<int>(my_num_blocks, partner_num_blocks);
            } else {
                merge_from_blocks = utils::range<int>(my_num_blocks, partner_num_blocks);
                merge_to_blocks = utils::range<int>(0, my_num_blocks);
            }
            std::vector<int> block_map = utils::range<int>(0, partition.blockmodel.getNum_blocks());
            for (int merge_from : merge_from_blocks) {
                best_merges[merge_from] = std::make_pair<int, double>(-1, std::numeric_limits<double>::max());
                for (int merge_to : merge_to_blocks) {
                    // Calculate the delta entropy given the current block assignment
                    EdgeWeights out_blocks = partition.blockmodel.blockmatrix()->outgoing_edges(merge_from);
                    EdgeWeights in_blocks = partition.blockmodel.blockmatrix()->incoming_edges(merge_from);
                    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
                    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
                    int k = k_out + k_in;
                    utils::ProposalAndEdgeCounts proposal{merge_to, k_out, k_in, k};
                    Delta delta = block_merge::blockmodel_delta(merge_from, proposal.proposal, partition.blockmodel);
                    int proposed_block_self_edges = partition.blockmodel.blockmatrix()->get(merge_to, merge_to)
                                                    + delta.get(merge_to, merge_to);
                    double dE = entropy::block_merge_delta_mdl(merge_from, proposal, partition.blockmodel, delta);
                    if (dE < best_merges[merge_from].second) {
                        best_merges[merge_from] = std::make_pair(merge_to, dE);
                        block_map[merge_from] = merge_to;
                    }
                }
            }
            std::vector<int> assignment = partition.blockmodel.block_assignment();
            for (int i = 0; i < subgraph.graph.num_vertices(); ++i) {
                assignment[i] = block_map[assignment[i]];
            }
            std::vector<int> mapping = Blockmodel::build_mapping(assignment);
            for (size_t i = 0; i < assignment.size(); ++i) {
                int block = assignment[i];
                int new_block = mapping[block];
                assignment[i] = new_block;
            }
            partition.blockmodel = Blockmodel((int) merge_to_blocks.size(), subgraph.graph, 0.5, assignment);
            std::cout << mpi.rank << " | merged Blockmodel to one with B = " << partition.blockmodel.getNum_blocks() << std::endl;
            bitmask <<= 1;
        } else {
            // The sender
            // Send number of vertices
            int local_num_vertices = subgraph.graph.num_vertices();
            std::vector<int> local_vertices = utils::constant<int>(subgraph.graph.num_vertices(), -1);
            std::vector<int> local_assignment = utils::constant<int>(subgraph.graph.num_vertices(), -1);
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(graph, subgraph, partition, local_vertices, local_assignment)
            for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
                int subgraph_index = subgraph.mapping[vertex];
                if (subgraph_index < 0) continue;  // vertex not present
                int assignment = partition.blockmodel.block_assignment(subgraph_index);
                local_vertices[subgraph_index] = vertex;
                local_assignment[subgraph_index] = assignment;
            }
            MPI_Send(&local_num_vertices, 1, MPI_INT, partner, NUM_VERTICES_TAG, MPI_COMM_WORLD);
            MPI_Send(local_vertices.data(), local_num_vertices, MPI_INT, partner, VERTICES_TAG, MPI_COMM_WORLD);
            MPI_Send(local_assignment.data(), local_num_vertices, MPI_INT, partner, BLOCKS_TAG, MPI_COMM_WORLD);
            done = true;
        }
        level++;
    }
    if (mpi.rank == 0) {
        // Finetune final assignment
        Blockmodel blockmodel = std::move(partition.blockmodel);
        blockmodel.setOverall_entropy(entropy::mdl(blockmodel, subgraph.graph.num_vertices(), subgraph.graph.num_edges()));
        BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        float iteration = sbp::get_intermediates().size();
        while (!sbp::done_blockmodeling(blockmodel, blockmodel_triplet)) {
            if (blockmodel.getNum_blocks_to_merge() != 0) {
                std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                          << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
            }
            double start_bm = MPI_Wtime();
            blockmodel = block_merge::merge_blocks(blockmodel, subgraph.graph, subgraph.graph.num_edges());
            block_merge::BlockMerge_time += MPI_Wtime() - start_bm;
            std::cout << "Starting MCMC vertex moves" << std::endl;
            double start_mcmc = MPI_Wtime();
//        if (args.algorithm == "async_gibbs_old" && iteration < float(args.asynciterations))
//            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
//        else
            common::candidates = std::uniform_int_distribution<int>(0, blockmodel.getNum_blocks() - 2);
            if (args.algorithm == "async_gibbs" && iteration < float(args.asynciterations))
                blockmodel = finetune::asynchronous_gibbs(blockmodel, subgraph.graph, blockmodel_triplet);
            else if (args.algorithm == "hybrid_mcmc")
                blockmodel = finetune::hybrid_mcmc(blockmodel, subgraph.graph, blockmodel_triplet);
            else // args.algorithm == "metropolis_hastings"
                blockmodel = finetune::metropolis_hastings(blockmodel, subgraph.graph, blockmodel_triplet);
            finetune::MCMC_time += MPI_Wtime() - start_mcmc;
            sbp::total_time += MPI_Wtime() - start_bm;
            sbp::add_intermediate(++iteration, subgraph.graph, -1, blockmodel.getOverall_entropy());
            blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
            common::candidates = std::uniform_int_distribution<int>(0, blockmodel.getNum_blocks() - 2);
        }
        // only last iteration result will calculate expensive modularity
        double modularity = -1;
        if (args.modularity)
            modularity = graph.modularity(blockmodel.block_assignment());
        sbp::add_intermediate(-1, subgraph.graph, modularity, blockmodel.getOverall_entropy());
        // Evaluate finetuned assignment
        // Reorder truth labels to match the reordered graph vertices
        std::vector<int> reordered_truth = utils::constant<int>(graph.num_vertices(), -1);
        for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            int mapped_index = subgraph.mapping[vertex];
            int truth_community = graph.assignment(vertex);
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
        std::cerr << "Sample size of " << args.samplesize << " is too low. Must be greater than 0.0" << std::endl;
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
