#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "dict_matrix.hpp"

#include "toy_example.hpp"

class DictMatrixTest : public ToyExample {
    void SetUp() override {
        ToyExample::ToySetUp(false);
    }
};

class DictMatrixComplexTest : public ComplexExample {
    void SetUp() override {
        ComplexExample::SetUp();
        ComplexToySetUp(false);
    }
};

TEST_F(DictMatrixTest, NeighborsAreCorrectlyReturned) {
    std::set<int> neighbors = B.blockmatrix()->neighbors(1);
    // neighbors should contain all 3 blocks
    EXPECT_EQ(neighbors.size(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(neighbors.find(i) != neighbors.end());
    }
}

TEST_F(DictMatrixTest, UpdateEdgeCountsIsCorrect) {
    B.blockmatrix()->update_edge_counts(Deltas);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int correct_val = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(B.blockmatrix()->get(row, col), correct_val);
        }
    }
}

TEST_F(DictMatrixComplexTest, NeighborsAreCorrectlyReturned) {
    std::set<int> neighbors = B.blockmatrix()->neighbors(1);
    B.print_blockmatrix();
    // neighbors should contain all 3 blocks
    std::vector<int> correct_neighbors = { 0, 3, 4, 5 };
    std::cout << "Correct neighbors: " << std::endl;
    utils::print<int>(correct_neighbors);
    std::cout << "Returned neighbors: " << std::endl;
    std::vector<int> n2(neighbors.begin(), neighbors.end());
    utils::print<int>(n2);
    EXPECT_EQ(neighbors.size(), correct_neighbors.size());
    for (const int neighbor : correct_neighbors) {
        EXPECT_TRUE(neighbors.find(neighbor) != neighbors.end());
    }
}

TEST_F(DictMatrixComplexTest, UpdateEdgeCountsIsCorrect) {
    B.blockmatrix()->update_edge_counts(Deltas);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int correct_val = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(B.blockmatrix()->get(row, col), correct_val);
        }
    }
}

