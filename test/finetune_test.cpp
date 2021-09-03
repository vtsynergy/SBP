#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "utils.hpp"

#include "toy_example.hpp"

// TODO: figure out correct placement of these
MPI_t mpi;  // Unused
Args args;  // Unused

class FinetuneTest : public ToyExample {
protected:
    common::ProposalAndEdgeCounts Proposal;
    EdgeCountUpdates Updates;
    PairIndexVector Delta;
    void SetUp() override {
        ToyExample::SetUp();
        Proposal = {0, 2, 3, 5};
        Updates.block_row = { 2, 1, 3 };
        Updates.block_col = { 1, 0, 3 };
        Updates.proposal_row = { 8, 1, 1 };
        Updates.proposal_col = { 8, 2, 2 };
        Delta[std::make_pair(0, 0)] = 1;
        Delta[std::make_pair(0, 1)] = 0;
        Delta[std::make_pair(0, 2)] = 1;
        Delta[std::make_pair(1, 0)] = 1;
        Delta[std::make_pair(1, 2)] = -1;
        Delta[std::make_pair(2, 0)] = 1;
        Delta[std::make_pair(2, 1)] = 0;
        Delta[std::make_pair(2, 2)] = -3;
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
            finetune::compute_delta_entropy(current_block, Proposal.proposal, B, graph.num_edges(), Updates,
                                  new_block_degrees);
    std::cout << "dE using updates = " << delta_entropy;
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
    PairIndexVector delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    EXPECT_EQ(delta.size(), 8) << "blockmodel deltas are the wrong size. Expected 6 but got " << delta.size();
    EXPECT_EQ(delta[std::make_pair(0, 0)], 1);
    EXPECT_EQ(delta[std::make_pair(0, 1)], 0);
    EXPECT_EQ(delta[std::make_pair(0, 2)], 1);
    EXPECT_EQ(delta[std::make_pair(1, 0)], 1);
    EXPECT_EQ(delta[std::make_pair(1, 2)], -1);
    EXPECT_EQ(delta[std::make_pair(2, 0)], 1);
    EXPECT_EQ(delta[std::make_pair(2, 1)], 0);
    EXPECT_EQ(delta[std::make_pair(2, 2)], -3);
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, BlockmodelDeltasShouldSumUpToZero) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    PairIndexVector delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    int sum = 0;
    for (const auto &entry : delta) {
        sum += entry.second;
    }
    EXPECT_EQ(sum, 0);
    vertex = 10;  // has a self-edge
    current_block = B.block_assignment(vertex);
    out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    delta = finetune::blockmodel_delta(vertex, current_block, Proposal.proposal, out_edges, in_edges, B);
    sum = 0;
    for (const auto &entry : delta) {
        sum += entry.second;
    }
    EXPECT_EQ(sum, 0);
}

TEST_F(FinetuneTest, BlockmodelDeltaGivesSameBlockmatrixAsEdgeCountUpdates) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    Blockmodel B1 = B.copy();
    B1.move_vertex(vertex, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    Blockmodel B2 = B.copy();
    B2.move_vertex(vertex, Proposal.proposal, Delta, new_block_degrees.block_degrees_out,
                   new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
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
    B.print_blockmodel();
    double E_before = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    double delta_entropy = finetune::compute_delta_entropy(B, Delta, new_block_degrees);
    B.move_vertex(vertex, Proposal.proposal, Delta, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    int blockmodel_edges = utils::sum<int>(B.blockmatrix()->values());
    EXPECT_EQ(blockmodel_edges, graph.num_edges()) << "edges in blockmodel = " << blockmodel_edges << " edges in graph = " << graph.num_edges();
    double E_after = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    B.print_blockmodel();
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy
            << " but actual dE was " << E_after << " - " << E_before << " = " << E_after - E_before;
}
