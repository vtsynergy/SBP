#include "sample.hpp"

#include "toy_example.hpp"

//MPI_t mpi;  // Unused
//Args args;  // Unused

class SampleTest : public ToyExample {
    void SetUp() override {
        ToyExample::SetUp();
        args.samplesize = 0.4;
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
}
