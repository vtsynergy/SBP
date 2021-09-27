#include <iostream>
//#include <limits.h>
#include <cmath>
#include <omp.h>
#include <vector>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "utils.hpp"

MPI_t mpi;
Args args;

std::vector<std::vector<int>> Graph1 { {0, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 1}, {3, 2}, {3, 5}, {4, 1}, {4, 6},
                                       {5, 4}, {5, 5}, {5, 6}, {5, 7}, {6, 4}, {7, 3}, {7, 9}, {8, 5}, {8, 7}, {9, 10},
                                       {10, 7}, {10, 8}, {10, 10}
};

std::vector<std::vector<int>> Graph2 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}
};

std::vector<std::vector<int>> Graph3 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}, {2, 5}, {4, 8}
};

std::vector<std::vector<int>> Graph4 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}, {2, 5}, {4, 8}, {0, 5}, {4, 6}
};

std::vector<std::vector<int>> Graph5 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}, {2, 5}, {4, 8}, {0, 5}, {4, 6}, {6, 0}, {5, 4}
};

std::vector<std::vector<int>> Graph6 { {0, 1}, {2, 0}, {0, 3}, {4, 0}, {5, 0}, {0, 6}, {2, 1}, {2, 3}, {5, 4}, {6, 5},
                                       {6, 1}, {3, 8}, {7, 9}, {8, 7}, {9, 8}
};

std::vector<std::vector<int>> Graph7 { {0, 1}, {2, 0}, {0, 3}, {4, 0}, {5, 0}, {0, 6}, {2, 1}, {2, 3}, {5, 4}, {6, 5},
                                       {6, 1}, {3, 8}, {7, 9}, {8, 7}, {9, 8}, {7, 3}
};

std::vector<double> conditional_distribution(Graph &graph, std::vector<int> &assignment, int vertex1) {
    std::vector<double> distribution(graph.num_vertices(), 0.0);
    for (int block = 0; block < graph.num_vertices(); ++block) {
        std::vector<int> modified_assignment(assignment);
        modified_assignment[vertex1] = block;
//        std::cout << "modified assignment: ";
//        utils::print<int>(modified_assignment);
        Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), 0.5, modified_assignment);
//        std::cout << "log_posterior_prob: " << blockmodel.log_posterior_probability() << " exp(log_p) = " << std::exp(blockmodel.log_posterior_probability()) << std::endl;
        distribution[block] = blockmodel.log_posterior_probability();
    }
    double min_log_posterior_probability = std::numeric_limits<double>::max();
    for (double val : distribution) {
        if (val < min_log_posterior_probability)
            min_log_posterior_probability = val;
    }
    min_log_posterior_probability = std::abs(min_log_posterior_probability);
    for (int block = 0; block < graph.num_vertices(); ++block) {
        distribution[block] = std::exp(min_log_posterior_probability + distribution[block]);
    }
//    std::cout << "distribution: ";
//    utils::print<double>(distribution);
    auto sum = utils::sum<double>(distribution);
    if (sum == 0)
        return distribution;
    distribution = distribution / sum;
    return distribution;
}


double total_variation_distance(Graph &graph, int vertex1, int vertex2, int block1, int block2) {
    std::vector<int> X = utils::range<int>(0, graph.num_vertices());  // graph.assignment());
    X[vertex2] = block1;
    std::vector<double> Xcd = conditional_distribution(graph, X, vertex1);
    std::vector<int> Y = utils::range<int>(0, graph.num_vertices());  // graph.assignment());
    Y[vertex2] = block2;
    std::vector<double> Ycd = conditional_distribution(graph, Y, vertex1);
    double tvd = 0.0;
    for (int block = 0; block < graph.num_vertices(); ++block) {
        tvd += std::abs(Xcd[block] - Ycd[block]);
    }
    tvd *= 0.5;
    return tvd;
}

double compute_influence(Graph &graph) {
    double influence = std::numeric_limits<double>::min();
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    for (int vertex1 = 0; vertex1 < graph.num_vertices(); ++vertex1) {
        double sum_influence_on_vertex1 = 0.0;
#pragma omp parallel for default(shared)
        for (int vertex2 = 0; vertex2 < graph.num_vertices(); ++vertex2) {
            if (vertex1 == vertex2) continue;
//            double max_influence_on_vertex1 = std::numeric_limits<double>::min();
//            // Keep block1 as the current block for vertex2
//            // Only loop over block2
//            int b1 = -1;
//            int b2 = -1;
//            for (int block1 = 0; block1 < graph.num_vertices(); ++block1) {
//                for (int block2 = block1 + 1; block2 < graph.num_vertices(); ++block2) {
//                    if (block1 == block2) continue;
            int block1 = vertex1;
            int block2 = vertex2;
            double max_influence_on_vertex1 = total_variation_distance(graph, vertex1, vertex2, block1, block2);
//                    double tvd = total_variation_distance(graph, vertex1, vertex2, block1, block2);
//                    if (tvd > max_influence_on_vertex1) {
//                        b1 = block1;
//                        b2 = block2;
//                    }
//                    max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
//                }
//            }
#pragma omp atomic
            sum_influence_on_vertex1 += max_influence_on_vertex1;
            influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
//            std::cout << "highest tvd on " << vertex2 << " --> " << vertex1 << " when b1 = " << b1 << " b2 = " << b2 << std::endl;
        }
        influence = std::max(influence, sum_influence_on_vertex1);
        std::cout << vertex1 << " (" << sum_influence_on_vertex1 << "): ";
        utils::print<double>(influence_matrix[vertex1]);
    }
    std::cout << "total influence = " << influence << std::endl;
    return influence;
}

double compute_influence(std::vector<std::vector<int>> &graph_edges) {
    int num_vertices = -1;
    int num_edges = (int) graph_edges.size();
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    for (const std::vector<int> &edge : graph_edges) {
        int from = edge[0];
        int to = edge[1];
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to, from);
        int max_v = std::max(from + 1, to + 1);
        num_vertices = std::max(max_v, num_vertices);
    }
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<int>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<int>());
    }
    std::vector<int> assignment = utils::range<int>(0, num_vertices);
    Graph graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
    return compute_influence(graph);
//    double influence = std::numeric_limits<double>::min();
//    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
//    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
//    for (int vertex1 = 0; vertex1 < graph.num_vertices(); ++vertex1) {
//        double sum_influence_on_vertex1 = 0.0;
//        for (int vertex2 = 0; vertex2 < graph.num_vertices(); ++vertex2) {
//            if (vertex1 == vertex2) continue;
//            double max_influence_on_vertex1 = std::numeric_limits<double>::min();
//            // Keep block1 as the current block for vertex2
//            // Only loop over block2
////            int b1 = -1;
////            int b2 = -1;
//            for (int block1 = 0; block1 < graph.num_vertices(); ++block1) {
//                for (int block2 = block1 + 1; block2 < graph.num_vertices(); ++block2) {
////                    if (block1 == block2) continue;
//                    double tvd = total_variation_distance(graph, vertex1, vertex2, block1, block2);
////                    if (tvd > max_influence_on_vertex1) {
////                        b1 = block1;
////                        b2 = block2;
////                    }
//                    max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
//                }
//            }
//            sum_influence_on_vertex1 += max_influence_on_vertex1;
//            influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
////            std::cout << "highest tvd on " << vertex2 << " --> " << vertex1 << " when b1 = " << b1 << " b2 = " << b2 << std::endl;
//        }
//        influence = std::max(influence, sum_influence_on_vertex1);
//        std::cout << vertex1 << " (" << sum_influence_on_vertex1 << "): ";
//        utils::print<double>(influence_matrix[vertex1]);
//    }
//    std::cout << "total influence = " << influence << std::endl;
//    return influence;
}

int main(int argc, char* argv[]) {
    args = Args(argc, argv);
    Graph graph = Graph::load(args);
//    compute_influence(Graph1);
//    compute_influence(Graph2);
//    compute_influence(Graph3);
//    compute_influence(Graph4);
//    compute_influence(Graph5);
//    compute_influence(Graph6);
//    compute_influence(Graph7);
    compute_influence(graph);
}