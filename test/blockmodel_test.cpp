#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "block_merge.hpp"

#include "toy_example.hpp"

class BlockmodelTest : public ToyExample {
protected:
    void SetUp() override {
        args.transpose = true;
        ToyExample::SetUp();
    }
};

class BlockmodelComplexTest : public ComplexExample {
protected:
    void SetUp() override {
        args.transpose = true;
        ComplexExample::SetUp();
    }
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

TEST_F(BlockmodelTest, MergeBlockIsCorrect) {
    // PREV
    // _ | 0 | 1 | 2
    // 0 | 7   1   0 | 8
    // 1 | 1   5   1 | 7
    // 2 | 1   1   6 | 8
    // UPDATED
    // _ | 0 | 1 | 2
    // 0 | 0   0   0 | 0
    // 1 | 0   14  1 | 15
    // 2 | 0   2   6 | 8
    //     0   16  7
//                       0  1  2  3  4  5  6  7  8  9  10
//        assignment = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2 };
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    B.merge_block(0, 1, delta);
    EXPECT_EQ(B.blockmatrix()->get(0, 0), 0);
    EXPECT_EQ(B.blockmatrix()->get(0, 1), 0);
    EXPECT_EQ(B.blockmatrix()->get(0, 2), 0);
    EXPECT_EQ(B.blockmatrix()->get(1, 0), 0);
    EXPECT_EQ(B.blockmatrix()->get(1, 1), 14);
    EXPECT_EQ(B.blockmatrix()->get(1, 2), 1);
    EXPECT_EQ(B.blockmatrix()->get(2, 0), 0);
    EXPECT_EQ(B.blockmatrix()->get(2, 1), 2);
    EXPECT_EQ(B.blockmatrix()->get(2, 2), 6);
    EXPECT_EQ(B.degrees_out(0), 0);
    EXPECT_EQ(B.degrees_out(1), 15);
    EXPECT_EQ(B.degrees_out(2), 8);
    EXPECT_EQ(B.degrees_in(0), 0);
    EXPECT_EQ(B.degrees_in(1), 16);
    EXPECT_EQ(B.degrees_in(2), 7);
    EXPECT_EQ(B.degrees(0), 0);
    EXPECT_EQ(B.degrees(1), 17);
    EXPECT_EQ(B.degrees(2), 9);
    for (int i = 0; i < B.block_assignment().size(); ++i) {
        if (i < 7) {
            EXPECT_EQ(B.block_assignment(i), 1);
        } else {
            EXPECT_EQ(B.block_assignment(i), 2);
        }
    }
}

TEST_F(BlockmodelTest, MergeBlockConsecutiveIsCorrect) {
//    // TODO
//    // PREV
//    // _ | 0 | 1 | 2
//    // 0 | 7   1   0 | 8
//    // 1 | 1   5   1 | 7
//    // 2 | 1   1   6 | 8
//    // UPDATED
//    // _ | 0 | 1 | 2
//    // 0 | 0   0   0 | 0
//    // 1 | 0   14  1 | 15
//    // 2 | 0   2   6 | 8
//    //     0   16  7
////                       0  1  2  3  4  5  6  7  8  9  10
////        assignment = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2 };
    std::vector<int> new_assignment = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    Blockmodel new_blockmodel(6, graph, 0.5, new_assignment);
//    int proposed_block = B.translate(1);
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    B.merge_block(0, 1, delta);
    std::cout << "After first merge: " << std::endl;
    B.print_blockmodel();
//    proposed_block = B.translate(0);
//    std::cout << "translated block = " << proposed_block << std::endl;
    delta = block_merge::blockmodel_delta(2, 0, B);
    std::cout << "computed second delta" << std::endl;
    B.merge_block(2, 0, delta);
    std::cout << "After second merge: " << std::endl;
    B.print_blockmodel();
    for (int row = 0; row < B.getNum_blocks(); ++row) {
    for (int col = 0; col < B.getNum_blocks(); ++col) {
    int val1 = B.blockmatrix()->get(row, col);
    int val2 = new_blockmodel.blockmatrix()->get(row, col);
    EXPECT_EQ(val1, val2)
    << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
    << " using assignment, value = " << val2;
    }
    EXPECT_EQ(B.block_assignment(row), new_blockmodel.block_assignment(row));
    EXPECT_EQ(B.degrees_out(row), new_blockmodel.degrees_out(row));
    EXPECT_EQ(B.degrees_in(row), new_blockmodel.degrees_in(row));
    EXPECT_EQ(B.degrees(row), new_blockmodel.degrees(row));
    }
    B.validate(graph);
//    Delta delta = block_merge::blockmodel_delta(0, 1, B);
//    B.merge_block(0, 1, delta);
//    EXPECT_EQ(B.blockmatrix()->get(0, 0), 0);
//    EXPECT_EQ(B.blockmatrix()->get(0, 1), 0);
//    EXPECT_EQ(B.blockmatrix()->get(0, 2), 0);
//    EXPECT_EQ(B.blockmatrix()->get(1, 0), 0);
//    EXPECT_EQ(B.blockmatrix()->get(1, 1), 14);
//    EXPECT_EQ(B.blockmatrix()->get(1, 2), 1);
//    EXPECT_EQ(B.blockmatrix()->get(2, 0), 0);
//    EXPECT_EQ(B.blockmatrix()->get(2, 1), 2);
//    EXPECT_EQ(B.blockmatrix()->get(2, 2), 6);
//    EXPECT_EQ(B.degrees_out(0), 0);
//    EXPECT_EQ(B.degrees_out(1), 15);
//    EXPECT_EQ(B.degrees_out(2), 8);
//    EXPECT_EQ(B.degrees_in(0), 0);
//    EXPECT_EQ(B.degrees_in(1), 16);
//    EXPECT_EQ(B.degrees_in(2), 7);
//    EXPECT_EQ(B.degrees(0), 0);
//    EXPECT_EQ(B.degrees(1), 17);
//    EXPECT_EQ(B.degrees(2), 9);
//    for (int i = 0; i < B.block_assignment().size(); ++i) {
//        if (i < 7) {
//            EXPECT_EQ(B.block_assignment(i), 1);
//        } else {
//            EXPECT_EQ(B.block_assignment(i), 2);
//        }
//    }
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

TEST_F(BlockmodelTest, MoveVertexWithBlockmodelDeltasDynamicBlockDegreesIsCorrect) {
    B.move_vertex(7, Deltas, Proposal);
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
    B.move_vertex(Move);
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
    B.move_vertex(SelfEdgeMove);
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

TEST_F(BlockmodelComplexTest, MergeBlockIsCorrect) {
    // TODO
    //  _ | 0 | 1 | 2 | 3 | 4 | 5 before
    //  0 | 4   1   0   0   0   0 | 5
    //  1 | 2   0   0   1   0   1 | 4
    //  2 | 1   0   0   1   0   0 | 2
    //  3 | 0   0   2   2   1   0 | 5
    //  4 | 0   2   0   0   0   0 | 2
    //  5 | 0   0   0   1   2   2 | 5
    //      7   3   2   5   3   3 | 23
    //  _ | 0 | 1 | 2 | 3 | 4 | 5 after
    //  0 | 4   1   0   0   0   0 | 5
    //  1 | 2   0   0   1   0   1 | 4
    //  2 | 1   0   0   1   0   0 | 2
    //  3 | 0   0   2   2   1   0 | 5
    //  4 | 0   2   0   0   0   0 | 2
    //  5 | 0   0   0   1   2   2 | 5
    //      7   3   2   5   3   3 | 23
    std::vector<int> new_assignment = { 1, 1, 1, 1, 2, 3, 3, 4, 5, 1, 5 };
    Blockmodel new_blockmodel(6, graph, 0.5, new_assignment);
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    B.merge_block(0, 1, delta);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = new_blockmodel.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
        EXPECT_EQ(B.block_assignment(row), new_blockmodel.block_assignment(row));
        EXPECT_EQ(B.degrees_out(row), new_blockmodel.degrees_out(row));
        EXPECT_EQ(B.degrees_in(row), new_blockmodel.degrees_in(row));
        EXPECT_EQ(B.degrees(row), new_blockmodel.degrees(row));
    }
    B.validate(graph);
}

//TEST_F(BlockmodelComplexTest, MergeBlockConsecutiveIsCorrect) {
//    // TODO
//    //  _ | 0 | 1 | 2 | 3 | 4 | 5 before
//    //  0 | 4   1   0   0   0   0 | 5
//    //  1 | 2   0   0   1   0   1 | 4
//    //  2 | 1   0   0   1   0   0 | 2
//    //  3 | 0   0   2   2   1   0 | 5
//    //  4 | 0   2   0   0   0   0 | 2
//    //  5 | 0   0   0   1   2   2 | 5
//    //      7   3   2   5   3   3 | 23
//    //  _ | 0 | 1 | 2 | 3 | 4 | 5 after
//    //  0 | 4   1   0   0   0   0 | 5
//    //  1 | 2   0   0   1   0   1 | 4
//    //  2 | 1   0   0   1   0   0 | 2
//    //  3 | 0   0   2   2   1   0 | 5
//    //  4 | 0   2   0   0   0   0 | 2
//    //  5 | 0   0   0   1   2   2 | 5
//    //      7   3   2   5   3   3 | 23
//    std::vector<int> new_assignment = { 1, 1, 1, 1, 2, 3, 3, 4, 5, 1, 5 };
//    Blockmodel new_blockmodel(6, graph, 0.5, new_assignment);
//    Delta delta = block_merge::blockmodel_delta(0, 1, B);
//    B.merge_block(0, 1, delta);
//    for (int row = 0; row < B.getNum_blocks(); ++row) {
//        for (int col = 0; col < B.getNum_blocks(); ++col) {
//            int val1 = B.blockmatrix()->get(row, col);
//            int val2 = new_blockmodel.blockmatrix()->get(row, col);
//            EXPECT_EQ(val1, val2)
//                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
//                                << " using assignment, value = " << val2;
//        }
//        EXPECT_EQ(B.block_assignment(row), new_blockmodel.block_assignment(row));
//        EXPECT_EQ(B.degrees_out(row), new_blockmodel.degrees_out(row));
//        EXPECT_EQ(B.degrees_in(row), new_blockmodel.degrees_in(row));
//        EXPECT_EQ(B.degrees(row), new_blockmodel.degrees(row));
//    }
//    B.validate(graph);
////    EXPECT_EQ(B.blockmatrix()->get(0, 0), 0);
////    EXPECT_EQ(B.blockmatrix()->get(0, 1), 0);
////    EXPECT_EQ(B.blockmatrix()->get(0, 2), 0);
////    EXPECT_EQ(B.blockmatrix()->get(1, 0), 0);
////    EXPECT_EQ(B.blockmatrix()->get(1, 1), 14);
////    EXPECT_EQ(B.blockmatrix()->get(1, 2), 1);
////    EXPECT_EQ(B.blockmatrix()->get(2, 0), 0);
////    EXPECT_EQ(B.blockmatrix()->get(2, 1), 2);
////    EXPECT_EQ(B.blockmatrix()->get(2, 2), 6);
////    EXPECT_EQ(B.degrees_out(0), 0);
////    EXPECT_EQ(B.degrees_out(1), 15);
////    EXPECT_EQ(B.degrees_out(2), 8);
////    EXPECT_EQ(B.degrees_in(0), 0);
////    EXPECT_EQ(B.degrees_in(1), 16);
////    EXPECT_EQ(B.degrees_in(2), 7);
////    EXPECT_EQ(B.degrees(0), 0);
////    EXPECT_EQ(B.degrees(1), 17);
////    EXPECT_EQ(B.degrees(2), 9);
////    for (int i = 0; i < B.block_assignment().size(); ++i) {
////        if (i < 7) {
////            EXPECT_EQ(B.block_assignment(i), 1);
////        } else {
////            EXPECT_EQ(B.block_assignment(i), 2);
////        }
////    }
//}

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
    B.move_vertex(Move);
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
    B.move_vertex(SelfEdgeMove);
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
