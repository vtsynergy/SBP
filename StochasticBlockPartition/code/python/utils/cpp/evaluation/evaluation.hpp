/***
 * Stores the current graph evaluation results.
 */
#ifndef CPPSBP_EVALUATION_HPP
#define CPPSBP_EVALUATION_HPP

// #include <limits>

// #include <Eigen/Core>
// #include <pybind11/eigen.h>
#include <pybind11/stl.h>

class Evaluation {
  public:
    Evaluation(py::dict args, int num_vertices, int num_edges) {

    }
  private:
    static std::vector<std::string> FIELD_NAMES {
        "block size variation",
        "block overlap",
        "streaming type",
        "num vertices",
        "num edges",
        "blocks retained (%)",
        "within to between edge ratio",
        "difference from exact uniform sample",
        "expansion quality",
        "subgraph clustering coefficient",
        "full graph clustering coefficient",
        "subgraph within to between edge ratio",
        "subgraph num blocks in algorithm partition",
        "subgraph num blocks in truth partition",
        "subgraph accuracy",
        "subgraph rand index",
        "subgraph adjusted rand index",
        "subgraph pairwise recall",
        "subgraph pairwise precision",
        "subgraph entropy of algorithm partition",
        "subgraph entropy of truth partition",
        "subgraph entropy of algorithm partition given truth partition",
        "subgraph entropy of truth partition given algorithm partition",
        "subgraph mutual information",
        "subgraph fraction of missed information",
        "subgraph fraction of erroneous information",
        "num block proposals",
        "beta",
        "sample size (%)",
        "sampling iterations",
        "sampling algorithm",
        "sparse",
        "delta entropy threshold",
        "nodal update threshold strategy",
        "nodal update threshold factor",
        "nodal update threshold direction",
        "num blocks in algorithm partition",
        "num blocks in truth partition",
        "accuracy",
        "rand index",
        "adjusted rand index",
        "pairwise recall",
        "pairwise precision",
        "entropy of algorithm partition",
        "entropy of truth partition",
        "entropy of algorithm partition given truth partition",
        "entropy of truth partition given algorithm partition",
        "mutual information",
        "fraction of missed information",
        "fraction of erroneous information",
        "num nodal updates",
        "num nodal update iterations",
        "num finetuning updates",
        "num finetuning iterations",
        "num iterations",
        "graph loading time",
        "sampling time",
        "total partition time",
        "total nodal update time",
        "total block merge time",
        "prepare the next iteration",
        "merging partitioned sample time",
        "cluster propagation time",
        "finetuning membership time"
    };
    static std::vector<std::string> DETAILS_FIELD_NAMES {
        "superstep",
        "step",
        "iteration",
        "substep",
        "time"
    };
    // CSV File Names
    std::string csv_file;
    std::string csv_details_file;
    // Dataset Parameters
    std::string block_size_variation;
    std::string block_verlap
    std::string streaming_type;
    int num_vertices;
    int num_edges;
    // Sampling Evaluation
    float blocks_retained = 0.0;
    float graph_edge_ratio = 0.0;
    float difference_from_exact_uniform_sample = 0.0;
    float expansion_quality = 0.0;
    float subgraph_clustering_coefficient = 0.0;
    float full_graph_clustering_coefficient = 0.0;
    float subgraph_edge_ratio = 0.0;
    float subgraph_num_blocks_algorithm = 0.0;
    float subgraph_num_blocks_truth = 0.0;
    float subgraph_accuracy = 0.0;
    float subgraph_rand_index = 0.0;
    float subgraph_adjusted_rand_index = 0.0;
}

// #include <sparse/boost_mapped_matrix.hpp>

// namespace py = pybind11;

// typedef py::EigenDRef<Eigen::Matrix<int, Eigen::Dynamic, 2>> Matrix2Column;
// typedef Eigen::VectorXi Vector;

// class Partition {
// public:
//     Partition(int num_blocks, std::vector<Matrix2Column> out_neighbors, float block_reduction_rate) {
//         this->num_blocks = num_blocks;
//         this->block_reduction_rate = block_reduction_rate;
//         this->overall_entropy = std::numeric_limits<float>::max();
//         this->blockmodel = BoostMappedMatrix(this->num_blocks, this->num_blocks);
//         // Set the block assignment
//         this->block_assignment.resize(this->num_blocks);
//         for (int i = 0; i < this->num_blocks; ++i) {
//             this->block_assignment(i) = i;
//         }
//         // Number of blocks to merge
//         this->num_blocks_to_merge = (int)(this->num_blocks * this->block_reduction_rate);
//         // Block degrees
//         this->block_degrees_out = Vector::Zero(this->num_blocks);
//         this->block_degrees_in = Vector::Zero(this->num_blocks);
//         this->block_degrees = Vector::Zero(this->num_blocks);
//         this->initialize_edge_counts(out_neighbors);
//     }
//     Partition(int num_blocks, std::vector<Matrix2Column> out_neighbors, float block_reduction_rate,
//               Vector block_assignment) {
//         this->num_blocks = num_blocks;
//         this->block_reduction_rate = block_reduction_rate;
//         this->overall_entropy = std::numeric_limits<float>::max();
//         this->blockmodel = BoostMappedMatrix(this->num_blocks, this->num_blocks);
//         // Set the block assignment
//         this->block_assignment = block_assignment;
//         // Number of blocks to merge
//         this->num_blocks_to_merge = (int)(this->num_blocks * this->block_reduction_rate);
//         // Block degrees
//         this->block_degrees_out = Vector::Zero(this->num_blocks);
//         this->block_degrees_in = Vector::Zero(this->num_blocks);
//         this->block_degrees = Vector::Zero(this->num_blocks);
//         this->initialize_edge_counts(out_neighbors);
//     }
//     void initialize_edge_counts(std::vector<Matrix2Column> neighbors);
//     Partition clone_with_true_block_membership(std::vector<Matrix2Column> neighbors, Vector true_block_membership);
//     Partition copy();
//     static Partition from_sample(int num_blocks, std::vector<Matrix2Column> neighbors, Vector sample_block_membership,
//                           std::map<int, int> mapping, float block_reduction_rate);
//     BoostMappedMatrix getBlockmodel() { return this->blockmodel; }
//     void setBlockmodel(BoostMappedMatrix blockmodel) { this->blockmodel = blockmodel; }
//     Vector getBlock_assignment() { return this->block_assignment; }
//     void setBlock_assignment(Vector block_assignment) { this->block_assignment = block_assignment; }
//     Vector getBlock_degrees() { return this->block_degrees; }
//     void setBlock_degrees(Vector block_degrees) { this->block_degrees = block_degrees; }
//     Vector getBlock_degrees_in() { return this->block_degrees_in; }
//     void setBlock_degrees_in(Vector block_degrees_in) { this->block_degrees_in = block_degrees_in; }
//     Vector getBlock_degrees_out() { return this->block_degrees_out; }
//     void setBlock_degrees_out(Vector block_degrees_out) { this->block_degrees_out = block_degrees_out; }
//     float getBlock_reduction_rate() { return this->block_reduction_rate; }
//     void setBlock_reduction_rate(float block_reduction_rate) { this->block_reduction_rate = block_reduction_rate; }
//     float getOverall_entropy() { return this->overall_entropy; }
//     void setOverall_entropy(float overall_entropy) { this->overall_entropy = overall_entropy; }
//     int getNum_blocks_to_merge() { return this->num_blocks_to_merge; }
//     void setNum_blocks_to_merge(int num_blocks_to_merge) { this->num_blocks_to_merge = num_blocks_to_merge; }
//     int getNum_blocks() { return this->num_blocks; }
//     void setNum_blocks(int num_blocks) { this->num_blocks = num_blocks; }
//     void merge_blocks(int from_block, int to_block);
//     void carry_out_best_merges(Eigen::VectorXd delta_entropy_for_each_block, Vector best_merge_for_each_block);

// private:
//     // Structure
//     int num_blocks;
//     BoostMappedMatrix blockmodel;
//     // Known info
//     Vector block_assignment;
//     Vector block_degrees;
//     Vector block_degrees_in;
//     Vector block_degrees_out;
//     float block_reduction_rate;
//     // Computed info
//     float overall_entropy;
//     int num_blocks_to_merge;
// };

#endif // CPPSBP_EVALUATION_HPP
