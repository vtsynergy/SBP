#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "blockmodel/sparse/delta.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "utils.hpp"

#include "toy_example.hpp"

// TODO: figure out correct placement of these
MPI_t mpi;  // Unused
Args args;  // Unused

class FinetuneTest : public ToyExample {
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

TEST_F(FinetuneTest, SetUpWorksCorrectly) {
    EXPECT_EQ(graph.num_vertices(), 11);
    EXPECT_EQ(graph.out_neighbors().size(), graph.num_vertices());
    EXPECT_EQ(graph.out_neighbors().size(), graph.in_neighbors().size());
    EXPECT_EQ(graph.num_edges(), 23);
}

TEST_F(FinetuneTest, OverallEntropyGivesCorrectAnswer) {
    double E = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, ENTROPY) << "Calculated entropy = " << E << " but was expecting " << ENTROPY;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, DenseDeltaEntropyGivesCorrectAnswer) {
    int vertex = 7;
    double E_before = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    int current_block = B.block_assignment(vertex);
    double delta_entropy =
            finetune::compute_delta_entropy(current_block, Proposal.proposal, B, graph.num_edges(), Updates, new_block_degrees);
    std::cout << "dE using updates = " << delta_entropy;
    B.move_vertex(vertex, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                           new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_after = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}

TEST_F(FinetuneTest, SparseEdgeCountUpdatesAreCorrect) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, true);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B, 7, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 2);
    EXPECT_EQ(updates.block_row[1], 1);
    EXPECT_EQ(updates.block_row[2], 3);
    EXPECT_EQ(updates.block_col[0], 1);
    EXPECT_EQ(updates.block_col[2], 3);
    EXPECT_EQ(updates.proposal_row[0], 8);
    EXPECT_EQ(updates.proposal_row[1], 1);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_col[0], 8);
    EXPECT_EQ(updates.proposal_col[1], 2);
    EXPECT_EQ(updates.proposal_col[2], 2);
}

TEST_F(FinetuneTest, SparseEdgeCountUpdatesWithSelfEdgesAreCorrect) {
    int vertex = 5;
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, true);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B, 5, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 1);
    EXPECT_EQ(updates.block_row[1], 2);
    EXPECT_EQ(updates.block_col[0], 2);
    EXPECT_EQ(updates.block_col[1], 2);
    EXPECT_EQ(updates.proposal_row[0], 9);
    EXPECT_EQ(updates.proposal_row[1], 2);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_col[0], 9);
    EXPECT_EQ(updates.proposal_col[1], 1);
    EXPECT_EQ(updates.proposal_col[2], 2);
}

TEST_F(FinetuneTest, SparseDeltaEntropyGivesCorrectAnswer) {
    int vertex = 7;
    double E_before = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    int current_block = B.block_assignment(vertex);
    double delta_entropy =
            finetune::compute_delta_entropy(current_block, Proposal.proposal, B, graph.num_edges(), SparseUpdates,
                                            new_block_degrees);
    std::cout << "dE using sparse updates = " << delta_entropy;
    B.move_vertex(vertex, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_after = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, BlockmodelDeltasAreCorrect) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    Delta delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    EXPECT_EQ(delta.entries().size(), 6) << "blockmodel deltas are the wrong size. Expected 6 but got " << delta.entries().size();
    EXPECT_EQ(delta.get(0,0), 1);
    EXPECT_EQ(delta.get(0,1), 0);
    EXPECT_EQ(delta.get(0,2), 1);
    EXPECT_EQ(delta.get(1,0), 1);
    EXPECT_EQ(delta.get(1,2), -1);
    EXPECT_EQ(delta.get(2,0), 1);
    EXPECT_EQ(delta.get(2,1), 0);
    EXPECT_EQ(delta.get(2,2), -3);
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, BlockmodelDeltasShouldSumUpToZero) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    Delta delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    int sum = 0;
    for (const auto &entry : delta.entries()) {
        sum += std::get<2>(entry);
    }
    EXPECT_EQ(sum, 0);
    vertex = 10;  // has a self-edge
    current_block = B.block_assignment(vertex);
    out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    sum = 0;
    for (const auto &entry : delta.entries()) {
        sum += std::get<2>(entry);
    }
    EXPECT_EQ(sum, 0);
}

TEST_F(FinetuneTest, BlockmodelDeltaGivesSameBlockmatrixAsEdgeCountUpdates) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    B.print_blockmatrix();
    Blockmodel B1 = B.copy();
    B1.move_vertex(vertex, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    B1.print_blockmatrix();
    Blockmodel B2 = B.copy();
    B2.move_vertex(vertex, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out,
                   new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    B2.print_blockmatrix();
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B1.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                << " using deltas, value = " << val2;
        }
    }
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, DeltaEntropyUsingBlockmodelDeltasGivesCorrectAnswer) {
    int vertex = 7;
    double E_before = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    double delta_entropy = finetune::compute_delta_entropy(B, Deltas, new_block_degrees);
    B.move_vertex(vertex, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    int blockmodel_edges = utils::sum<int>(B.blockmatrix()->values());
    EXPECT_EQ(blockmodel_edges, graph.num_edges()) << "edges in blockmodel = " << blockmodel_edges << " edges in graph = " << graph.num_edges();
    double E_after = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy
            << " but actual dE was " << E_after << " - " << E_before << " = " << E_after - E_before;
}

TEST_F(FinetuneTest, HastingsCorrectionBlockCountsAreTheSameWithAndWithoutBlockmodelDeltas) {
    int vertex = 7;
    std::unordered_map<int, int> block_counts1;
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
    std::unordered_map<int, int> block_counts2;
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

TEST_F(FinetuneTest, HastingsCorrectionWithAndWithoutDeltaGivesSameResult) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    double hastings1 = finetune::hastings_correction(vertex, graph, B, Deltas, current_block, Proposal, new_block_degrees);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    double hastings2 = finetune::hastings_correction(B, blocks_out_neighbors, blocks_in_neighbors, Proposal, Updates, new_block_degrees);
    EXPECT_FLOAT_EQ(hastings1, hastings2);
}

TEST_F(FinetuneTest, SpecialCaseGivesCorrectSparseEdgeCountUpdates) {
    int vertex = 6;
    int current_block = B3.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, true);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B3, vertex, current_block, 0, out_edges, in_edges, updates);
    EXPECT_EQ(updates.block_row[0], 1);
    EXPECT_EQ(updates.block_row[1], 0);
    EXPECT_EQ(updates.block_row[2], 1);
    EXPECT_EQ(updates.block_row[3], 1);
    EXPECT_EQ(updates.block_row[4], 1);
    EXPECT_EQ(updates.block_row[5], 0);
    EXPECT_EQ(updates.block_col[0], 0);
    EXPECT_EQ(updates.block_col[1], 1);
    EXPECT_EQ(updates.block_col[2], 0);
    EXPECT_EQ(updates.block_col[3], 1);
    EXPECT_EQ(updates.block_col[4], 0);
    EXPECT_EQ(updates.block_col[5], 1);
    EXPECT_EQ(updates.proposal_row[0], 4);
    EXPECT_EQ(updates.proposal_row[1], 1);
    EXPECT_EQ(updates.proposal_row[2], 1);
    EXPECT_EQ(updates.proposal_row[3], 0);
    EXPECT_EQ(updates.proposal_row[4], 0);
    EXPECT_EQ(updates.proposal_row[5], 0);
    EXPECT_EQ(updates.proposal_col[0], 4);
    EXPECT_EQ(updates.proposal_col[1], 2);
    EXPECT_EQ(updates.proposal_col[2], 2);
    EXPECT_EQ(updates.proposal_col[3], 1);
    EXPECT_EQ(updates.proposal_col[4], 0);
    EXPECT_EQ(updates.proposal_col[5], 0);
}

TEST_F(FinetuneTest, SpecialCaseShouldGiveCorrectDeltaEntropy) {
    int vertex = 6;
    common::ProposalAndEdgeCounts proposal { 0, 1, 2, 3 };
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
    double E_before = finetune::overall_entropy(B3, 11, 23);
    double dE = finetune::overall_entropy(B5, 11, 23) - E_before;
    EXPECT_FLOAT_EQ(dE, result.delta_entropy);
}
