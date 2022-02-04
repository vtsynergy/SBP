#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "blockmodel/sparse/delta.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "utils.hpp"

#include "toy_example.hpp"
#include "typedefs.hpp"

// TODO: figure out correct placement of these
//MPI_t mpi;  // Unused
//Args args;  // Unused

class EntropyTest : public ToyExample {
protected:
    Blockmodel B3;
    void SetUp() override {
        ToyExample::SetUp();
        std::vector<int> assignment3 = { 0, 0, 0, 1, 2, 3, 3, 4, 5, 1, 5 };
        B3 = Blockmodel(6, graph.out_neighbors(), 0.5, assignment3);
//        Deltas[std::make_pair(0, 0)] = 1;
//        Deltas[std::make_pair(0, 1)] = 0;
//        Deltas[std::make_pair(0, 2)] = 1;
//        Deltas[std::make_pair(1, 0)] = 1;
//        Deltas[std::make_pair(1, 2)] = -1;
//        Deltas[std::make_pair(2, 0)] = 1;
//        Deltas[std::make_pair(2, 1)] = 0;
//        Deltas[std::make_pair(2, 2)] = -3;
    }
};

TEST_F(EntropyTest, SetUpWorksCorrectly) {
    EXPECT_EQ(graph.num_vertices(), 11);
    EXPECT_EQ(graph.out_neighbors().size(), graph.num_vertices());
    EXPECT_EQ(graph.out_neighbors().size(), graph.in_neighbors().size());
    EXPECT_EQ(graph.num_edges(), 23);
}

TEST_F(EntropyTest, OverallEntropyGivesCorrectAnswer) {
    double E = entropy::mdl(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, ENTROPY) << "Calculated entropy = " << E << " but was expecting " << ENTROPY;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(EntropyTest, DenseDeltaEntropyGivesCorrectAnswer) {
    int vertex = 7;
    double E_before = entropy::mdl(B, graph.num_vertices(), graph.num_edges());
    int current_block = B.block_assignment(vertex);
    double delta_entropy =
            finetune::compute_delta_entropy(current_block, Proposal.proposal, B, graph.num_edges(), Updates, new_block_degrees);
    std::cout << "dE using updates = " << delta_entropy;
    B.move_vertex(vertex, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                           new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_after = entropy::mdl(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}

TEST_F(EntropyTest, SparseDeltaEntropyGivesCorrectAnswer) {
    int vertex = 7;
    double E_before = entropy::mdl(B, graph.num_vertices(), graph.num_edges());
    int current_block = B.block_assignment(vertex);
    double delta_entropy =
            finetune::compute_delta_entropy(current_block, Proposal.proposal, B, graph.num_edges(), SparseUpdates,
                                            new_block_degrees);
    std::cout << "dE using sparse updates = " << delta_entropy;
    B.move_vertex(vertex, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_after = entropy::mdl(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(EntropyTest, DeltaEntropyUsingBlockmodelDeltasGivesCorrectAnswer) {
    int vertex = 7;
    double E_before = entropy::mdl(B, graph.num_vertices(), graph.num_edges());
    double delta_entropy = finetune::compute_delta_entropy(B, Deltas, Proposal);
    B.move_vertex(vertex, Deltas, Proposal);
    int blockmodel_edges = utils::sum<int>(B.blockmatrix()->values());
    EXPECT_EQ(blockmodel_edges, graph.num_edges()) << "edges in blockmodel = " << blockmodel_edges << " edges in graph = " << graph.num_edges();
    double E_after = entropy::mdl(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy
            << " but actual dE was " << E_after << " - " << E_before << " = " << E_after - E_before;
}

TEST_F(EntropyTest, HastingsCorrectionBlockCountsAreTheSameWithAndWithoutBlockmodelDeltas) {
    int vertex = 7;
    MapVector<int> block_counts1;
//    std::unordered_map<int, int> block_counts1;
    for (const int neighbor : graph.out_neighbors(vertex)) {
        int neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    for (const int neighbor : graph.in_neighbors(vertex)) {
        if (neighbor == vertex) continue;
        int neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    utils::print(block_counts1);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    MapVector<int> block_counts2;
//    std::unordered_map<int, int> block_counts2;
    for (uint i = 0; i < blocks_out_neighbors.indices.size(); ++i) {
        int block = blocks_out_neighbors.indices[i];
        int weight = blocks_out_neighbors.values[i];
        block_counts2[block] += weight; // block_count[new block] should initialize to 0
    }
    for (uint i = 0; i < blocks_in_neighbors.indices.size(); ++i) {
        int block = blocks_in_neighbors.indices[i];
        int weight = blocks_in_neighbors.values[i];
        block_counts2[block] += weight; // block_count[new block] should initialize to 0
    }
    utils::print(block_counts2);
    for (const auto entry : block_counts1) {
        EXPECT_EQ(entry.second, block_counts2[entry.first]);
    }
    for (const auto entry : block_counts2) {
        EXPECT_EQ(entry.second, block_counts1[entry.first]);
    }
}

TEST_F(EntropyTest, HastingsCorrectionWithAndWithoutDeltaGivesSameResult) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    double hastings1 = finetune::hastings_correction(vertex, graph, B, Deltas, current_block, Proposal);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    double hastings2 = finetune::hastings_correction(B, blocks_out_neighbors, blocks_in_neighbors, Proposal, Updates, new_block_degrees);
    EXPECT_FLOAT_EQ(hastings1, hastings2);
}

TEST_F(EntropyTest, SpecialCaseShouldGiveCorrectDeltaEntropy) {
    int vertex = 6;
    utils::ProposalAndEdgeCounts proposal {0, 1, 2, 3 };
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, true);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B3, vertex, 3, 0, out_edges, in_edges, updates);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            3, B3, 1, 4, proposal);
    Blockmodel B4 = B3.copy();
    Blockmodel B5 = B3.copy();
    finetune::VertexMove result = finetune::move_vertex_nodelta(6, 3, proposal, B4, graph, out_edges, in_edges);
    B5.move_vertex(vertex, 3, 0, updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_before = entropy::mdl(B3, 11, 23);
    double dE = entropy::mdl(B5, 11, 23) - E_before;
    std::cout << "======== Before move ========" << std::endl;
    B3.print_blockmodel();
    std::cout << "======== After move =======" << std::endl;
    B5.print_blockmodel();
    EXPECT_FLOAT_EQ(dE, result.delta_entropy);
}
