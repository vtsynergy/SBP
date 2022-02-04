#include <vector>

#include <gtest/gtest.h>

#include "entropy.hpp"
#include "finetune.hpp"  // for mdl()
#include "block_merge.hpp"
#include "blockmodel/sparse/delta.hpp"

#include "toy_example.hpp"

class BlockMergeTest : public ToyExample {
    void SetUp() override {
        ToyExample::SetUp();
        Updates = EdgeCountUpdates();
        Updates.block_row = { 0, 0, 0 };
        Updates.block_col = { 0, 0, 0 };
        Updates.proposal_row = { 0, 14, 1 };
        Updates.proposal_col = { 0, 14, 2 };
        SparseUpdates = SparseEdgeCountUpdates();
        SparseUpdates.proposal_row[1] = 14;
        SparseUpdates.proposal_row[2] = 1;
        SparseUpdates.proposal_col[1] = 14;
        SparseUpdates.proposal_col[2] = 2;
        Deltas = Delta(0, 1);
        Deltas.add(0, 0, -7);
        Deltas.add(0, 1, -1);
        Deltas.add(1, 0, -1);
        Deltas.add(1, 1, 9);
        Deltas.add(2, 0, -1);
        Deltas.add(2, 1, 1);
        new_block_degrees.block_degrees_out = { 0, 15, 8 };
        new_block_degrees.block_degrees_in = { 0, 16, 7 };
        new_block_degrees.block_degrees = { 0, 17, 9 };
        std::vector<int> assignment2 = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2 };
        B2 = Blockmodel(3, graph.out_neighbors(), 0.5, assignment2);
    }
};

TEST_F(BlockMergeTest, BlockmodelDeltaIsCorrectlyComputed) {
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}

TEST_F(BlockMergeTest, BlockmodelDeltaEntropyIsCorrectlyComputeWithDenseUpdates) {
    double E_before = entropy::mdl(B, 11, 23);
    double dE = block_merge::compute_delta_entropy(0, 1, 23, B, Updates, new_block_degrees);
    double E_after = entropy::mdl(B2, 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}

TEST_F(BlockMergeTest, BlockmodelDeltaEntropyIsCorrectlyComputeWithSparseUpdates) {
    double E_before = entropy::mdl(B, 11, 23);
    double dE = block_merge::compute_delta_entropy_sparse(0, 1, 23, B, SparseUpdates, new_block_degrees);
    double E_after = entropy::mdl(B2, 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}

TEST_F(BlockMergeTest, BlockmodelDeltaEntropyIsCorrectlyComputeWithBlockmodelDeltas) {
    double E_before = entropy::mdl(B, 11, 23);
    double dE = block_merge::compute_delta_entropy_sparse(0, B, Deltas, new_block_degrees);
    double E_after = entropy::mdl(B2, 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}

TEST_F(BlockMergeTest, BlockmodelDeltaEntropyIsCorrectlyComputeWithBlockmodelDeltasSansBlockDegrees) {
    double E_before = entropy::mdl(B, 11, 23);
    double dE = block_merge::compute_delta_entropy(0, {1, B.degrees_out(0),
                                                       B.degrees_in(0), B.degrees(0)}, B, Deltas);
    double E_after = entropy::mdl(B2, 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}
