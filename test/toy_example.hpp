#ifndef DISTRIBUTEDSBP_TEST_TOY_EXAMPLE_H
#define DISTRIBUTEDSBP_TEST_TOY_EXAMPLE_H

#include <vector>

#include <gtest/gtest.h>

#include "args.hpp"
#include "blockmodel.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "utils.hpp"

const double ENTROPY = 92.58797747;

class ToyExample : public ::testing::Test {
protected:
    // My variables
    std::vector<int> assignment;
    Blockmodel B;
    Graph graph;
    void SetUp() override {
        std::vector<std::vector<int>> edges {
                {0, 0},
                {0, 1},
                {0, 2},
                {1, 2},
                {2, 3},
                {3, 1},
                {3, 2},
                {3, 5},
                {4, 1},
                {4, 6},
                {5, 4},
                {5, 5},
                {5, 6},
                {5, 7},
                {6, 4},
                {7, 3},
                {7, 9},
                {8, 5},
                {8, 7},
                {9, 10},
                {10, 7},
                {10, 8},
                {10, 10}
        };
        int num_vertices = 11;
        int num_edges = (int) edges.size();
        assignment = { 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 };
        NeighborList out_neighbors;
        NeighborList in_neighbors;
        for (const std::vector<int> &edge : edges) {
            int from = edge[0];
            int to = edge[1];
            utils::insert_nodup(out_neighbors, from , to);
            utils::insert_nodup(in_neighbors, to, from);
        }
        graph = Graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
        B = Blockmodel(3, graph.out_neighbors(), 0.5, assignment);
    }
//    virtual void TearDown() {
//
//    }
};

#endif //DISTRIBUTEDSBP_TEST_TOY_EXAMPLE_H
