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
};

TEST_F(FinetuneTest, SetUpWorksCorrectly) {
    EXPECT_EQ(graph.num_vertices(), 11);
    EXPECT_EQ(graph.out_neighbors().size(), graph.num_vertices());
    EXPECT_EQ(graph.out_neighbors().size(), graph.in_neighbors().size());
    EXPECT_EQ(graph.num_edges(), 23);
}

TEST_F(FinetuneTest, OverallEntropyGivesCorrectAnswer) {
    Blockmodel B = Blockmodel(3, graph.out_neighbors(), 0.5, assignment);
    double E = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, ENTROPY) << "Calculated entropy = " << E << " but was expecting " << ENTROPY;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, DenseDeltaEntropyGivesCorrectAnswer) {
    int vertex = 7;
    Blockmodel B = Blockmodel(3, graph.out_neighbors(), 0.5, assignment);
    double E_before = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    common::ProposalAndEdgeCounts proposal {0, 2, 3, 5};
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    int self_edge_weight = 0;
    for (uint i = 0; i < out_edges.indices.size(); ++i) {
        if (out_edges.indices[i] == vertex) {
            self_edge_weight = out_edges.values[i];
            break;
        }
    }
    EXPECT_EQ(self_edge_weight, 0) << "self edge weight was not " << 0;
    EdgeCountUpdates updates = finetune::edge_count_updates(B.blockmatrix(), current_block, proposal.proposal,
                                                            blocks_out_neighbors, blocks_in_neighbors,
                                                            self_edge_weight);
    int current_block_self_edges = B.blockmatrix()->get(current_block, current_block)
                                   + updates.block_row[current_block];
    int proposed_block_self_edges = B.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                    + updates.proposal_row[proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, proposal);

    double delta_entropy =
            finetune::compute_delta_entropy(current_block, proposal.proposal, B, graph.num_edges(), updates,
                                  new_block_degrees);
    B.move_vertex(vertex, current_block, proposal.proposal, updates, new_block_degrees.block_degrees_out,
                           new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_after = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, BlockmodelDeltasAreCorrect) {
    int vertex = 7;
    Blockmodel B = Blockmodel(3, graph.out_neighbors(), 0.5, assignment);
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    common::ProposalAndEdgeCounts proposal {0, 2, 3, 5};
    PairIndexVector delta = finetune::blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, B);
    EXPECT_EQ(delta.size(), 6) << "blockmodel deltas are the wrong size. Expected 6 but got " << delta.size();
    EXPECT_EQ(delta[std::make_pair(0, 0)], 1);
    EXPECT_EQ(delta[std::make_pair(0, 2)], 1);
    EXPECT_EQ(delta[std::make_pair(1, 0)], 1);
    EXPECT_EQ(delta[std::make_pair(1, 2)], -1);
    EXPECT_EQ(delta[std::make_pair(2, 0)], 1);
    EXPECT_EQ(delta[std::make_pair(2, 2)], -3);
}

/// TODO: same test but using a vertex with a self edge
TEST_F(FinetuneTest, BlockmodelDeltasShouldSumUpToZero) {
    int vertex = 7;
    Blockmodel B = Blockmodel(3, graph.out_neighbors(), 0.5, assignment);
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    common::ProposalAndEdgeCounts proposal {0, 2, 3, 5};
    PairIndexVector delta = finetune::blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, B);
    int sum = 0;
    for (const auto &entry : delta) {
        sum += entry.second;
    }
    EXPECT_EQ(sum, 0);
    vertex = 10;  // has a self-edge
    current_block = B.block_assignment(vertex);
    out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    proposal = {0, 2, 3, 5};
    delta = finetune::blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, B);
    sum = 0;
    for (const auto &entry : delta) {
        sum += entry.second;
    }
    EXPECT_EQ(sum, 0);
}

TEST_F(FinetuneTest, BlockmodelDeltaGivesSameBlockmatrixAsEdgeCountUpdates) {
    int vertex = 7;
    Blockmodel B = Blockmodel(3, graph.out_neighbors(), 0.5, assignment);
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    common::ProposalAndEdgeCounts proposal {0, 2, 3, 5};
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    int self_edge_weight = 0;
    for (uint i = 0; i < out_edges.indices.size(); ++i) {
        if (out_edges.indices[i] == vertex) {
            self_edge_weight = out_edges.values[i];
            break;
        }
    }
    EXPECT_EQ(self_edge_weight, 0) << "self edge weight was not " << 0;
    EdgeCountUpdates updates = finetune::edge_count_updates(B.blockmatrix(), current_block, proposal.proposal,
                                                            blocks_out_neighbors, blocks_in_neighbors,
                                                            self_edge_weight);
    int current_block_self_edges = B.blockmatrix()->get(current_block, current_block)
                                   + updates.block_row[current_block];
    int proposed_block_self_edges = B.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                    + updates.proposal_row[proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, proposal);
    Blockmodel B1 = B.copy();
    B1.move_vertex(vertex, current_block, proposal.proposal, updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    PairIndexVector delta = finetune::blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, B);
    Blockmodel B2 = B.copy();
    B2.move_vertex(vertex, proposal.proposal, delta, new_block_degrees.block_degrees_out,
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
    Blockmodel B = Blockmodel(3, graph.out_neighbors(), 0.5, assignment);
    double E_before = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    int current_block = B.block_assignment(vertex);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, false);
    common::ProposalAndEdgeCounts proposal {0, 2, 3, 5};
    PairIndexVector delta = finetune::blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, B);
    int current_block_self_edges = B.blockmatrix()->get(current_block, current_block)
                                   + get(delta, std::make_pair(current_block, current_block));
    int proposed_block_self_edges = B.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                    + get(delta, std::make_pair(proposal.proposal, proposal.proposal));
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, proposal);
    double delta_entropy = finetune::compute_delta_entropy(B, delta, new_block_degrees);
    B.move_vertex(vertex, proposal.proposal, delta, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    int blockmodel_edges = utils::sum<int>(B.blockmatrix()->values());
    EXPECT_EQ(blockmodel_edges, graph.num_edges()) << "edges in blockmodel = " << blockmodel_edges << " edges in graph = " << graph.num_edges();
    double E_after = finetune::overall_entropy(B, graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}
