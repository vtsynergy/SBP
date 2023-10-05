#include "graph/undirected_graph.hpp"

#include <execution>
#include "mpi.h"

#include "utils.hpp"
#include "mpi_data.hpp"

void UndirectedGraph::add_edge(long from, long to) {
    utils::insert_nodup(this->_out_neighbors, from , to);
    if (from == to) {
        this->_self_edges[from] = true;
    } else {
        utils::insert_nodup(this->_out_neighbors, to, from);
    }
    this->_num_edges++;
}

std::vector<long> UndirectedGraph::degrees() const {
    std::vector<long> vertex_degrees;
    for (long vertex = 0; vertex < this->_num_vertices; ++vertex) {
        vertex_degrees.push_back(long(this->_out_neighbors[vertex].size()));
    }
    return vertex_degrees;
}

//double UndirectedGraph::modularity(const std::vector<long> &assignment) const {
//    // See equation for Q in: https://hal.archives-ouvertes.fr/hal-01231784/document
//    double result = 0.0;
//    for (long vertex_i = 0; vertex_i < this->_num_vertices; ++vertex_i) {
//        for (long vertex_j = 0; vertex_j < this->_num_vertices; ++vertex_j) {
//            if (assignment[vertex_i] != assignment[vertex_j]) continue;
//            double edge_weight = 0.0;
//            for (long neighbor : this->_out_neighbors[vertex_i]) {
//                if (neighbor == vertex_j) {
//                    edge_weight = 1.0;
//                    break;
//                }
//            }
//            long deg_out_i = long(this->_out_neighbors[vertex_i].size());
//            long deg_in_j = long(this->_out_neighbors[vertex_j].size());
//            double temp = edge_weight - (double(deg_out_i * deg_in_j) / (2.0 * double(this->_num_edges)));
//            result += temp;
//        }
//    }
//    result /= (2.0 * double(this->_num_edges));
//    return result;
//}

std::vector<long> UndirectedGraph::neighbors(long vertex) const {
    return this->_out_neighbors[vertex];
}

// TODO: move to GraphInterface?
void UndirectedGraph::sort_vertices() {
    if (args.degreeproductsort) {
        this->degree_product_sort();
        return;
    }
//    std::cout << "Starting to sort vertices" << std::endl;
//    double start_t = MPI_Wtime();
    std::vector<long> vertex_degrees = this->degrees();
    std::vector<int> indices = utils::range<int>(0, this->_num_vertices);
    std::nth_element(std::execution::par_unseq, indices.data(), indices.data() + int(args.mh_percent * this->_num_vertices),
                     indices.data() + indices.size(), [&vertex_degrees](size_t i1, size_t i2) {
                return vertex_degrees[i1] > vertex_degrees[i2];
            });
    // std::sort(std::execution::par_unseq, indices.data(), indices.data() + indices.size(),  // sort in descending order
    //           [vertex_degrees](size_t i1, size_t i2) { return vertex_degrees[i1] > vertex_degrees[i2]; });
    for (int index = 0; index < this->_num_vertices; ++index) {
        int vertex = indices[index];
        if (index < (args.mh_percent * this->_num_vertices)) {
//            std::cout << "high degree vertex: " << vertex << " degree = " << vertex_degrees[vertex] << std::endl;
            this->_high_degree_vertices.push_back(vertex);
        } else {
//            std::cout << "low degree vertex: " << vertex << " degree = " << vertex_degrees[vertex] << std::endl;
            this->_low_degree_vertices.push_back(vertex);
        }
    }
//    std::cout << "Done sorting vertices, time = " << MPI_Wtime() - start_t << "s" << std::endl;
//    std::cout << "Range = " << *std::min_element(vertex_degrees.begin(), vertex_degrees.end()) << " - " << *std::max_element(vertex_degrees.begin(), vertex_degrees.end()) << std::endl;
    int num_islands = 0;
    for (int deg : vertex_degrees) {
        if (deg == 0) num_islands++;
    }
    std::cout << "Num island vertices = " << num_islands << std::endl;
}

// TODO: move to GraphInterface?
//void UndirectedGraph::degree_product_sort() {
////    std::cout << "Starting to sort vertices based on influence" << std::endl;
////    double start_t = MPI_Wtime();
//    std::vector<std::pair<std::pair<long, long>, long>> edge_info = this->sorted_edge_list();
//    MapVector<bool> selected;
//    int num_to_select = int(args.mh_percent * this->_num_vertices);
//    int edge_index = 0;
//    while (long(selected.size()) < num_to_select) {
//        const std::pair<std::pair<long, long>, long> &edge = edge_info[edge_index];
//        selected[edge.first.first] = true;
//        selected[edge.first.second] = true;
//        edge_index++;
//    }
//    for (const std::pair<long, bool> &entry : selected) {
//        this->_high_degree_vertices.push_back(entry.first);
//    }
//    for (long vertex = 0; vertex < this->_num_vertices; ++vertex) {
//        if (selected[vertex]) continue;
//        this->_low_degree_vertices.push_back(vertex);
//    }
////    std::cout << "Done sorting vertices, time = " << MPI_Wtime() - start_t << "s" << std::endl;
//}

// TODO: move to GraphInterface?
//long UndirectedGraph::num_islands() const {
//    std::vector<long> vertex_degrees = this->degrees();
//    long num_islands = 0;
//    for (const long &degree : vertex_degrees) {
//        if (degree == 0) num_islands++;
//    }
//    return num_islands;
//}

//std::vector<std::pair<std::pair<long, long>, long>> UndirectedGraph::sorted_edge_list() const {
//    std::vector<long> vertex_degrees = this->degrees();
//    std::vector<std::pair<std::pair<long, long>, long>> edge_info;
//    for (long source = 0; source < this->_num_vertices; ++source) {
//        const std::vector<long> &neighbors = this->_out_neighbors[source];
//        for (const long &dest : neighbors) {
//            long information = vertex_degrees[source] * vertex_degrees[dest];
//            edge_info.emplace_back(std::make_pair(source, dest), information);
//        }
//    }
//    std::sort(std::execution::par_unseq, edge_info.begin(), edge_info.end(), [](const auto &i1, const auto &i2) {
//        return i1.second > i2.second;
//    });
//    return edge_info;
//}
