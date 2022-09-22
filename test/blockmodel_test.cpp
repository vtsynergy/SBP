#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"

#include "toy_example.hpp"

class BlockmodelTest : public ToyExample {
};

class BlockmodelComplexTest : public ComplexExample {
//protected:
//    VertexMove_v2 Move {
//            -0.01,  // random change in entropy value
//            true,
//            6,
//            Proposal.proposal,
//            EdgeWeights { { 4 }, { 1 } },
//            EdgeWeights { { 4, 5 }, { 1, 1 }}
//    };
//    VertexMove_v2 SelfEdgeMove {
//        -0.01,
//        true,
//        5,
//        0,
//        EdgeWeights { { 4, 5, 6, 7 }, { 1, 1, 1, 1 } },
//        EdgeWeights { { 3, 8 }, { 1, 1 }}
//    };
//    std::vector<int> assignment3 = { 0, 0, 0, 1, 2, 0, 3, 4, 5, 1, 5 };
//    Blockmodel B3 = Blockmodel(6, graph, 0.5, assignment3);
};

TEST_F(BlockmodelTest, BlockDegreesAreCorrectlyInstantiated) {
    EXPECT_EQ(B.degrees_out(0), 8);
    EXPECT_EQ(B.degrees_out(1), 7);
    EXPECT_EQ(B.degrees_out(2), 8);
    EXPECT_EQ(B.degrees_in(0), 9);
    EXPECT_EQ(B.degrees_in(1), 7);
    EXPECT_EQ(B.degrees_in(2), 7);
    EXPECT_EQ(B.degrees(0), 10);
    EXPECT_EQ(B.degrees(1), 9);
    EXPECT_EQ(B.degrees(2), 9);
}

TEST_F(BlockmodelTest, MoveVertexWithDenseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(7, 2, Proposal.proposal, Updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithSparseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(7, 2, Proposal.proposal, SparseUpdates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithBlockmodelDeltasIsCorrect) {
    B.move_vertex(7, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithVertexEdgesIsCorrect) {
    B.move_vertex(7, B.block_assignment(7), Move);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithSelfEdgesUsingVertexEdgesIsCorrect) {
    std::cout << "Blockmatrix before move: " << std::endl;
    B.print_blockmatrix();
    B.move_vertex(5, B.block_assignment(5), SelfEdgeMove);
    std::cout << "Blockmatrix after move: " << std::endl;
    B.print_blockmatrix();
    std::cout << "Actual blockmatrix: " << std::endl;
    B3.print_blockmatrix();
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B3.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithDenseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(6, 3, Proposal.proposal, Updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithSparseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(6, 3, Proposal.proposal, SparseUpdates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithBlockmodelDeltasIsCorrect) {
    B.move_vertex(6, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithBlockmodelDeltasAndOnTheFlyBlockDegreesIsCorrect) {
    B.move_vertex(6, Deltas, Proposal);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithVertexEdgesIsCorrect) {
    std::cout << "Blockmatrix before move: " << std::endl;
    B.print_blockmatrix();
    B.move_vertex(6, B.block_assignment(6), Move);
    std::cout << "Blockmatrix after move: " << std::endl;
    B.print_blockmatrix();
    std::cout << "Actual blockmatrix: " << std::endl;
    B2.print_blockmatrix();
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithSelfEdgesUsingVertexEdgesIsCorrect) {
    B.move_vertex(5, B.block_assignment(5), SelfEdgeMove);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B3.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}
