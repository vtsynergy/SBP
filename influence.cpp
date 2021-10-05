#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#include <iostream>
#include <fstream>
//#include <limits.h>
#include <cmath>
#include <omp.h>
#include <vector>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "block_merge.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "sbp.hpp"
#include "utils.hpp"

MPI_t mpi;
Args args;

struct result_t {
    std::vector<std::pair<double, double>> csv_row;
    double f1;
    double mdl;
} Result;

double stop = false;

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

std::vector<double> conditional_distribution(const Graph &graph, const std::vector<int> &assignment, int vertex1, int num_blocks = -1, bool mdl = false) {
    if (num_blocks == -1)
        num_blocks = graph.num_vertices();
    std::vector<double> distribution(num_blocks, 0.0);
    for (int block = 0; block < num_blocks; ++block) {
        std::vector<int> modified_assignment(assignment);
        modified_assignment[vertex1] = block;
        std::vector<int> block_map(num_blocks, -1);
        int block_number = 0;
        for (int i = 0; i < graph.num_vertices(); ++i) {
            int _block = modified_assignment[i];
            if (block_map[_block] == -1) {
                block_map[_block] = block_number;
                block_number++;
            }
            modified_assignment[i] = block_map[_block];
        }
//        std::cout << "modified assignment: ";
//        utils::print<int>(modified_assignment);
        Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), 0.5, modified_assignment);
//        std::cout << "log_posterior_prob: " << blockmodel.log_posterior_probability() << " exp(log_p) = " << std::exp(blockmodel.log_posterior_probability()) << std::endl;
        if (mdl)
            distribution[block] = 1.0 / finetune::overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges());
        else  // use log posterior probability
            distribution[block] = blockmodel.log_posterior_probability();
    }
    if (!mdl) {
        double min_log_posterior_probability = std::numeric_limits<double>::max();
        for (double val : distribution) {
            if (val < min_log_posterior_probability)
                min_log_posterior_probability = val;
        }
        min_log_posterior_probability = std::abs(min_log_posterior_probability);
        for (int block = 0; block < num_blocks; ++block) {
            distribution[block] = std::exp(min_log_posterior_probability + distribution[block]);
        }
    }
//    std::cout << "distribution: ";
//    utils::print<double>(distribution);
    auto sum = utils::sum<double>(distribution);
    if (sum == 0)
        return distribution;
    distribution = distribution / sum;
    return distribution;
}

double total_variation_distance(const Blockmodel &B, const Graph &graph, int vertex1, int vertex2, int block1, int block2, bool mdl = false) {
    std::vector<int> X(B.block_assignment());  // graph.assignment());
    X[vertex2] = block1;
    std::vector<double> Xcd = conditional_distribution(graph, X, vertex1, B.getNum_blocks(), mdl);
    std::vector<int> Y(B.block_assignment());  // graph.assignment());
    Y[vertex2] = block2;
    std::vector<double> Ycd = conditional_distribution(graph, Y, vertex1, B.getNum_blocks(), mdl);
    double tvd = 0.0;
    for (int block = 0; block < B.getNum_blocks(); ++block) {
//        if (vertex1 == 0 && vertex2 == 7) {
//            std::cout << "block = " << block << "|" << Xcd[block] << " - " << Ycd[block] << "| = " << Xcd[block] - Ycd[block] << std::endl;
//        }
        tvd += std::abs(Xcd[block] - Ycd[block]);
    }
//    if (vertex1 == 0 && vertex2 == 7) exit(0);
    tvd *= 0.5;
    return tvd;
}

std::pair<double,double> compute_influence(const Graph &graph, const Blockmodel &B, bool do_merge = false, bool mdl = false) {
//    std::vector<int> initial_assignment = utils::range<int>(0, graph.num_vertices());
//    Blockmodel B(graph.num_vertices(), graph.out_neighbors(), 0.5, initial_assignment);
//    if (do_merge)
//        B = block_merge::merge_blocks(B, graph.out_neighbors(), graph.num_edges());
//    std::cout << "Block assignment = ";
//    utils::print<int>(B.block_assignment());
    std::vector<double> influence(graph.num_vertices(), std::numeric_limits<double>::min());
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    for (int vertex1 = 0; vertex1 < graph.num_vertices(); ++vertex1) {
        double sum_influence_on_vertex1 = 0.0;
#pragma omp parallel for default(shared)
        for (int vertex2 = 0; vertex2 < graph.num_vertices(); ++vertex2) {
            if (vertex1 == vertex2) continue;
            double max_influence_on_vertex1 = std::numeric_limits<double>::min();
            // Keep block1 as the current block for vertex2
            // Only loop over b1lock2
//            int b1 = -1;
//            int b2 = -1;
            for (int block1 = 0; block1 < B.getNum_blocks(); ++block1) {
                for (int block2 = block1 + 1; block2 < B.getNum_blocks(); ++block2) {
                    if (block1 == block2) continue;
                    double tvd = total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
//                    if (tvd > max_influence_on_vertex1) {
//                        b1 = block1;
//                        b2 = block2;
//                    }
                    max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
                }
            }
#pragma omp atomic
            sum_influence_on_vertex1 += max_influence_on_vertex1;
            influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
//            std::cout << "highest tvd on " << vertex2 << " --> " << vertex1 << " when b1 = " << b1 << " b2 = " << b2 << std::endl;
        }
        influence[vertex1] = sum_influence_on_vertex1;
//        max_influence = std::max(max_influence, sum_influence_on_vertex1);
//        int max_influence_vertex = utils::argmax<double>(influence_matrix[vertex1]);
//        std::cout << vertex1 << " (" << sum_influence_on_vertex1 << "," << max_influence_vertex << "): ";
//        utils::print<double>(influence_matrix[vertex1]);
    }
    double max_influence = std::numeric_limits<double>::min();
    double avg_influence = 0.0;
    for (double alpha : influence) {
        max_influence = std::max(alpha, max_influence);
        avg_influence += alpha;
    }
    avg_influence /= double(graph.num_vertices());
    std::cout << "total (max) influence = " << max_influence << " avg influence = " << avg_influence << std::endl;
    return std::make_pair(max_influence, avg_influence);
}

Graph to_graph(const std::vector<std::vector<int>> &graph_edges) {
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
    return graph;
}

//std::pair<double, double> compute_influence(const std::vector<std::vector<int>> &graph_edges, bool do_merge = false, bool mdl = false) {
//    int num_vertices = -1;
//    int num_edges = (int) graph_edges.size();
//    NeighborList out_neighbors;
//    NeighborList in_neighbors;
//    for (const std::vector<int> &edge : graph_edges) {
//        int from = edge[0];
//        int to = edge[1];
//        utils::insert_nodup(out_neighbors, from , to);
//        utils::insert_nodup(in_neighbors, to, from);
//        int max_v = std::max(from + 1, to + 1);
//        num_vertices = std::max(max_v, num_vertices);
//    }
//    while (out_neighbors.size() < size_t(num_vertices)) {
//        out_neighbors.push_back(std::vector<int>());
//    }
//    while (in_neighbors.size() < size_t(num_vertices)) {
//        in_neighbors.push_back(std::vector<int>());
//    }
//    std::vector<int> assignment = utils::range<int>(0, num_vertices);
//    Graph graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
//    return compute_influence(graph, do_merge, mdl);
//}

void print_csv(const std::vector<std::pair<double,double>>& csv_row, double mdl, double f1, const std::string& tag = "test") {
    std::ostringstream filepath_stream;
    filepath_stream << "./influence_results/" << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.overlap << "_" << args.blocksizevar << ".csv";
    std::string filepath = filepath_stream.str();
    bool write_header = false;
    if (!fs::exists(filepath)) {
        fs::create_directories(filepath_dir);
        write_header = true;
    }
    std::ofstream file;
    file.open(filepath, std::ios_base::app);  // , std::ofstream::out);
    if (write_header) {
        file << "tag,numvertices,overlap,blocksizevar,undirected,algorithm";
        for (int i = 0; i < 5; ++i) {
            file << ",itr" << i << "_ll_max_influence,itr" << i << "_ll_avg_influence,itr" << i << "_mdl_max_influence,itr" << i << "_mdl_avg_influence";
            if (i == 0)
                file << ",itr0.5_ll_max_influence,it0.5_ll_avg_influence,itr0.5_mdl_max_influence,itr0.5_mdl_avg_influence";
        }
        file << ",mdl,f1" << std::endl;
    }
    file << tag << "," << args.numvertices << "," << args.overlap << "," << args.blocksizevar << "," << args.undirected;
    file << "," << args.algorithm;
    for (int i = 0; i < 5; ++i) {
        auto res = csv_row[i*2];
        file << "," << res.first << "," << res.second;
        res = csv_row[i*2 + 1];
        file << "," << res.first << "," << res.second;
    }
    file << "," << mdl << "," << f1 << std::endl;
    file.close();
}

void stochastic_block_partition(Graph &graph, const std::string &tag = "test") {
    std::vector<std::pair<double, double>> csv_row;  // log-likelihood influence, mdl-influence at 0, 0.5, 1, 2, 3, ... iterations
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), 0.5);
    csv_row.push_back(compute_influence(graph, blockmodel, false, false));
    csv_row.push_back(compute_influence(graph, blockmodel, false, true));
    std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
              << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    int iteration = 0;
    while (!sbp::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::merge_blocks(blockmodel, graph.out_neighbors(), graph.num_edges());
        if (iteration == 0) {
            csv_row.push_back(compute_influence(graph, blockmodel, false, false));
            csv_row.push_back(compute_influence(graph, blockmodel, false, true));
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else  // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        iteration++;
        if (iteration < 5) {
            csv_row.push_back(compute_influence(graph, blockmodel, false, false));
            csv_row.push_back(compute_influence(graph, blockmodel, false, true));
        }
    }
    double mdl = finetune::overall_entropy(blockmodel, graph.num_vertices(), graph.num_edges());
    double f1 = evaluate::evaluate_blockmodel(graph, blockmodel);
    print_csv(csv_row, mdl, f1, tag);
}

int main(int argc, char* argv[]) {
    args = Args(argc, argv);
//    Graph graph = Graph::load(args);
    Graph G = to_graph(Graph1);
    stochastic_block_partition(G, args.tag);
//    compute_influence(Graph1);
//    compute_influence(Graph2);
//    compute_influence(Graph3);
//    compute_influence(Graph4);
//    compute_influence(Graph5);
//    compute_influence(Graph6);
//    compute_influence(Graph7);
//    compute_influence(graph);
//    compute_influence(Graph1, true);
//    compute_influence(Graph2, true);
//    compute_influence(Graph3, true);
//    compute_influence(Graph4, true);
//    compute_influence(Graph5, true);
//    compute_influence(Graph6, true);
//    compute_influence(Graph7, true);
//    compute_influence(graph, true);
//    compute_influence(Graph1, false, true);
//    compute_influence(Graph2, false, true);
//    compute_influence(Graph3, false, true);
//    compute_influence(Graph4, false, true);
//    compute_influence(Graph5, false, true);
//    compute_influence(Graph6, false, true);
//    compute_influence(Graph7, false, true);
//    compute_influence(graph, false, true);
//    compute_influence(Graph1, true, true);
//    compute_influence(Graph2, true, true);
//    compute_influence(Graph3, true, true);
//    compute_influence(Graph4, true, true);
//    compute_influence(Graph5, true, true);
//    compute_influence(Graph6, true, true);
//    compute_influence(Graph7, true, true);
//    compute_influence(graph, true, true);
}
#pragma clang diagnostic pop