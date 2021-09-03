#include <vector>

#include <gtest/gtest.h>

#include "common.hpp"

#include "toy_example.hpp"

class CommonTest : public ToyExample {
};

TEST_F(CommonTest, NewBlockDegreesAreCorrectlyComputed) {
    int vertex = 7;
    int current_block = B.block_assignment(vertex);
    common::ProposalAndEdgeCounts proposal {0, 2, 3, 5};
    int current_block_self_edges = 3;
    int proposed_block_self_edges = 8;
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, proposal);
    EXPECT_EQ(new_block_degrees.block_degrees_out[0], 10);
    EXPECT_EQ(new_block_degrees.block_degrees_out[1], 7);
    EXPECT_EQ(new_block_degrees.block_degrees_out[2], 6);
    EXPECT_EQ(new_block_degrees.block_degrees_in[0], 12);
    EXPECT_EQ(new_block_degrees.block_degrees_in[1], 7);
    EXPECT_EQ(new_block_degrees.block_degrees_in[2], 4);
    // TODO: when computing new_block_degrees, fix error where block_degrees are improperly calculated (k != k_in + k_out if there are self_edges)
    // TODO: same error may be present in blockmodel creation and updates
    EXPECT_EQ(new_block_degrees.block_degrees[0], 14);
    EXPECT_EQ(new_block_degrees.block_degrees[1], 9);
    EXPECT_EQ(new_block_degrees.block_degrees[2], 7);
}

// TODO: new test to make sure proposal has the correct value for `num_neighbor_edges`
