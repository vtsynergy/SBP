#include <vector>

#include <gtest/gtest.h>

#include "entropy.hpp"
#include "block_merge.hpp"
#include "blockmodel/sparse/delta.hpp"

#include "toy_example.hpp"

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

TEST_F(BlockMergeTest, BlockMergeDeltaMDLIsCorrectlyComputedAfterConsecutiveMerges) {
    double E_before = entropy::mdl(B, 11, 23);
    Delta d = block_merge::blockmodel_delta(0, 1, B);
    utils::ProposalAndEdgeCounts proposal { 1, B.degrees_out(0), B.degrees_in(0), B.degrees(0) };
    double dE = entropy::block_merge_delta_mdl(0, proposal, B, d);
    B.merge_block(0, 1, d);
    std::vector<int> merged_assignment = B.block_assignment();
    Blockmodel mergedB = Blockmodel(3, graph, 0.5, merged_assignment);
    double E_after = entropy::mdl(mergedB, 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
    E_before = E_after;
    d = block_merge::blockmodel_delta(2, 0, B);
    proposal = { 0, B.degrees_out(2), B.degrees_in(2), B.degrees(2) };
    dE = entropy::block_merge_delta_mdl(2, proposal, B, d);
    B.merge_block(2, 0, d);
    merged_assignment = B.block_assignment();
    mergedB = Blockmodel(3, graph, 0.5, merged_assignment);
    E_after = entropy::mdl(mergedB, 11, 23);
    EXPECT_TRUE(abs(dE - (E_after - E_before)) < 0.0001);
}
