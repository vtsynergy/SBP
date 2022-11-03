/**
 * The distributed finetuning phase of the stochastic block blockmodeling algorithm.
 */
#ifndef SBP_DIST_FINETUNE_HPP
#define SBP_DIST_FINETUNE_HPP

#include "distributed/dist_finetune.hpp"

#include "distributed/dist_common.hpp"
#include "entropy.hpp"
#include "finetune.hpp"

namespace finetune::dist {

//int MCMC_iterations = 0;
//double MCMC_time = 0.0;
//double MCMC_sequential_time = 0.0;
//double MCMC_parallel_time = 0.0;
//double MCMC_vertex_move_time = 0.0;
//uint MCMC_moves = 0;
//int num_surrounded = 0;
std::ofstream my_file;

MPI_Datatype Membership_t;

std::vector<Membership> mpi_get_assignment_updates(const std::vector<Membership> &membership_updates) {
    int num_moves = (int) membership_updates.size();
    int rank_moves[mpi.num_processes];
//    double t1 = MPI_Wtime();
//    my_file << mpi.rank << "," << MCMC_iterations << "," << t1 - t0 << std::endl;
    MPI_Allgather(&num_moves, 1, MPI_INT, &rank_moves, 1, MPI_INT, MPI_COMM_WORLD);
    int offsets[mpi.num_processes];
    offsets[0] = 0;
    for (int i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i - 1] + rank_moves[i - 1];
    }
    int batch_vertex_moves = offsets[mpi.num_processes - 1] + rank_moves[mpi.num_processes - 1];
    std::vector<Membership> collected_membership_updates(batch_vertex_moves);
    MPI_Allgatherv(membership_updates.data(), num_moves, Membership_t, collected_membership_updates.data(),
                   rank_moves, offsets, Membership_t, MPI_COMM_WORLD);
    return collected_membership_updates;
}

TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    my_file.open(args.csv, std::ios::out | std::ios::app);
//    MPI_Datatype Membership_t;
    int membership_blocklengths[2] = {1, 1};
    MPI_Aint membership_displacements[2] = {0, sizeof(int)};
    MPI_Datatype membership_types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, membership_blocklengths, membership_displacements, membership_types, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    double old_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
//    double initial_entropy = blockmodel.getOverall_entropy();
    double new_entropy = 0;
//    double t0;
//    double t1;
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double num_batches = args.batches;
        int batch_size = int(ceil(graph.num_vertices() / num_batches));
        // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
        // asynchronous Gibbs sampling
        std::vector<int> block_assignment(blockmodel.block_assignment());
//        int my_vertices = 0;
        for (int batch = 0; batch < graph.num_vertices() / batch_size; ++batch) {
//            t0 = MPI_Wtime();
            int start = batch * batch_size;
            int end = std::min(graph.num_vertices(), (batch + 1) * batch_size);
            std::vector<Membership> membership_updates;
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(start, end, blockmodel, graph, vertex_moves, membership_updates, block_assignment)
            for (int vertex = start; vertex < end; ++vertex) {
                // TODO: separate "new" code so can be switched on/off
                // TODO: batch by % of my vertices? Can be calculated same time as load balancing
                if (!blockmodel.owns_vertex(vertex)) continue;
                VertexMove proposal = dist::propose_gibbs_move(blockmodel, vertex, graph);
                if (proposal.did_move) {
//                    assert(blockmodel.stores(proposal.proposed_block));  // assert no work with default(none) until gcc 9.3.0
                    #pragma omp critical (updates)
                    {
                        membership_updates.push_back(Membership{vertex, proposal.proposed_block});
                    }
                }
            }
            std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
            // END MPI COMMUNICATION
            int batch_vertex_moves = 0;
            for (const Membership &membership: collected_membership_updates) {
                if (block_assignment[membership.vertex] != membership.block) batch_vertex_moves++;
                block_assignment[membership.vertex] = membership.block;
            }
            blockmodel.set_block_assignment(block_assignment);
            blockmodel.build_two_hop_blockmodel(graph.out_neighbors());
            blockmodel.initialize_edge_counts(graph);
            vertex_moves += batch_vertex_moves;
        }
        new_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, blockmodels, new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    my_file.close();
    // are there more iterations with the 2-hop blockmodel due to restricted vertex moves?
    return blockmodel;
}

// TODO: implement this!
TwoHopBlockmodel &hybrid_mcmc(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    my_file.open(args.csv, std::ios::out | std::ios::app);
    int membership_blocklengths[2] = {1, 1};  // Number of items in each field of Membership_t
    MPI_Aint membership_displacements[2] = {0, sizeof(int)};
    MPI_Datatype membership_types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, membership_blocklengths, membership_displacements, membership_types, &Membership_t);
    MPI_Type_commit(&Membership_t);
//    std::cout << "Distributed Hybrid MCMC iteration" << std::endl;
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    double old_entropy = 0.0;
    std::vector<double> delta_entropies;
//    std::vector<int> vertex_moves;
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
//    double initial_entropy = blockmodel.getOverall_entropy();
    double num_batches = args.batches;
    int num_low_degree_vertices = int(graph.low_degree_vertices().size());
    int batch_size = int(ceil(num_low_degree_vertices / num_batches));
    std::vector<Membership> membership_updates;
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        std::cout << "thread_limit: " << omp_get_max_threads() << std::endl;
        int _vertex_moves = 0;
        double start_t = MPI_Wtime();
        for (int vertex : graph.high_degree_vertices()) {  // Only run Metropolis-Hastings on high-degree vertices
            if (!blockmodel.owns_vertex(vertex)) continue;
            VertexMove proposal = dist::propose_mh_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                membership_updates.push_back(Membership{vertex, proposal.proposed_block});
            }
        }
        double sequential_t = MPI_Wtime();
        MCMC_sequential_time += sequential_t - start_t;
//        assert(blockmodel.validate(graph));
        for (int batch = 0; batch < num_low_degree_vertices / batch_size; ++batch) {
            start_t = MPI_Wtime();
            int start = batch * batch_size;
            int end = std::min(num_low_degree_vertices, (batch + 1) * batch_size);
            // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
            // asynchronous Gibbs sampling
            std::vector<int> block_assignment(blockmodel.block_assignment());
            std::vector<VertexMove_v2> moves(graph.num_vertices());
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(start, end, blockmodel, graph, _vertex_moves, block_assignment, membership_updates)
            for (int index = start; index < end; ++index) {
                int vertex = graph.low_degree_vertices()[index];
                VertexMove proposal = dist::propose_gibbs_move(blockmodel, vertex, graph);
                if (proposal.did_move) {
                    #pragma omp critical (updates)
                    {
                        membership_updates.push_back(Membership{vertex, proposal.proposed_block});
                    }
                }
            }
            std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
            for (const Membership &membership : collected_membership_updates) {
                if (block_assignment[membership.vertex] != membership.block) _vertex_moves++;
                block_assignment[membership.vertex] = membership.block;
            }
            blockmodel.set_block_assignment(block_assignment);
            blockmodel.build_two_hop_blockmodel(graph.out_neighbors());
            blockmodel.initialize_edge_counts(graph);
            double parallel_t = MPI_Wtime();
            MCMC_parallel_time += parallel_t - start_t;
            for (const VertexMove_v2 &move : moves) {
                if (!move.did_move) continue;
//                int current_block = blockmodel.block_assignment(move.vertex);
                blockmodel.move_vertex(move);
            }
            MCMC_vertex_move_time += MPI_Wtime() - parallel_t;
//            assert(blockmodel.validate(graph));
        }
        double new_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);;
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << _vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += _vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, blockmodels, new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    MCMC_moves += total_vertex_moves;
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

TwoHopBlockmodel &metropolis_hastings(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    // std::cout << "running distributed metropolis hastings yo!" << std::endl;
    my_file.open(args.csv, std::ios::out | std::ios::app);
    int membership_blocklengths[2] = {1, 1};  // Number of items in each field of Membership_t
    MPI_Aint membership_displacements[2] = {0, sizeof(int)};
    MPI_Datatype membership_types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, membership_blocklengths, membership_displacements, membership_types, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    double old_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
//    double initial_entropy = blockmodel.getOverall_entropy();
    double new_entropy = 0;
//    double t0, t1;
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
//        t0 = MPI_Wtime();
        // Block assignment used to re-create the Blockmodel after each iteration to incorporate moves from other ranks
        std::vector<int> block_assignment(blockmodel.block_assignment());
        std::vector<Membership> membership_updates;
        int vertex_moves = 0;
        for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            if (!blockmodel.owns_vertex(vertex)) continue;
            VertexMove proposal = dist::propose_mh_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                assert(blockmodel.stores(proposal.proposed_block));
                membership_updates.push_back(Membership{vertex, proposal.proposed_block});
            }
        }
        std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
        for (const Membership &membership: collected_membership_updates) {
            block_assignment[membership.vertex] = membership.block;
        }
        blockmodel.set_block_assignment(block_assignment);
        blockmodel.build_two_hop_blockmodel(graph.out_neighbors());
        blockmodel.initialize_edge_counts(graph);
        vertex_moves += collected_membership_updates.size();
        new_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        if (early_stop(iteration, blockmodels, new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    my_file.close();
    return blockmodel;
}

bool early_stop(int iteration, DistBlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies) {
    size_t last_index = delta_entropies.size() - 1;
    if (delta_entropies[last_index] == 0.0) {
        return true;
    }
    if (iteration < 3) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (blockmodels.get(2).empty) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return average < threshold;
}

VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);

    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex, true);

    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    return eval_vertex_move(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);  // getBlock_assignment()[vertex];
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex, true);

//    blockmodel.validate(graph);
    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    return move_vertex(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

}  // namespace finetune::dist

#endif // SBP_DIST_FINETUNE_HPP