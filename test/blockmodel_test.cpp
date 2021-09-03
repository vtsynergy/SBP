#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"

#include "toy_example.hpp"

class BlockmodelTest : public ToyExample {
};

TEST_F(BlockmodelTest, BlockDegreesAreCorrectlyInstantiated) {
    EXPECT_EQ(B.getBlock_degrees_out()[0], 8);
    EXPECT_EQ(B.getBlock_degrees_out()[1], 7);
    EXPECT_EQ(B.getBlock_degrees_out()[2], 8);
    EXPECT_EQ(B.getBlock_degrees_in()[0], 9);
    EXPECT_EQ(B.getBlock_degrees_in()[1], 7);
    EXPECT_EQ(B.getBlock_degrees_in()[2], 7);
    EXPECT_EQ(B.getBlock_degrees()[0], 10);
    EXPECT_EQ(B.getBlock_degrees()[1], 9);
    EXPECT_EQ(B.getBlock_degrees()[2], 9);
}
