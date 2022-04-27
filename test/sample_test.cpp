#include "sample.hpp"

#include "toy_example.hpp"

//MPI_t mpi;  // Unused
//Args args;  // Unused

class SampleTest : public ToyExample {
public:
    std::vector<std::pair<int, int>> sample_edges;
    std::vector<int> sample_assignment;
    void SetUp() override {
        ToyExample::SetUp();
        args.samplesize = 0.4;
        sample_edges = {
                { 0, 0 },
                { 0, 2 },
                { 1, 0 },
                { 1, 3 },
                { 2, 1 },
        };
        assignment = { 1, 0, 2, 0 };
    }
};

TEST_F(SampleTest, MaxDegreeSamplingIsCorrect) {
    sample::Sample s = sample::max_degree(graph);
    EXPECT_EQ(s.graph.num_vertices(), 4);
    for (int v = 0; v < s.graph.num_vertices(); ++v) {
        std::cout << "v = " << v << ": ";
        utils::print<int>(s.graph.out_neighbors(v));
    }
    EXPECT_EQ(s.graph.num_edges(), 5);
    EXPECT_EQ(s.mapping.size(), graph.num_vertices());
    for (const std::pair<int, int> &edge : sample_edges) {
        int from = edge.first;
        int to = edge.second;
        int found = false;
        for (int neighbor : s.graph.out_neighbors(from)) {
            if (neighbor == to) {
                found = true;
                continue;
            }
        }
        EXPECT_TRUE(found);
    }
    for (int i = 0; i < graph.num_vertices(); ++i) {
        int sample_vertex = s.mapping[i];
        if (sample_vertex == -1) continue;
        EXPECT_EQ(graph.assignment(i), s.graph.assignment(sample_vertex));
    }
}
