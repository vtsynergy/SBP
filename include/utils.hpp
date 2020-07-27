/***
 * Utility functions.
 */
#ifndef SBP_UTILS_HPP
#define SBP_UTILS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"

typedef std::vector<std::vector<int>> NeighborList;

namespace utils {

/// Builds the base path for the graph and true assignment .tsv files.
/// Assumes the file is saved in the following directory:
/// <args.directory>/<args.type>/<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar
/// Assumes the graph file is named:
/// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_nodes.tsv
/// Assumes the true assignmnet file is named:
/// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_truePartition.tsv
std::string build_filepath(argparse::ArgumentParser &args);

/// Assumes filepath corresponds to the path of a CSV file, and reads it as such.
/// All data stored as strings.
/// Note: does NOT differentiate between header row and data rows, and does NOT do data type conversion.
std::vector<std::vector<std::string>> read_csv(std::filesystem::path &filepath);

/// Inserts the given edge into the neighbors list. Assumes the graph is unweighted.
void insert(NeighborList &neighbors, int from, int to);

}

#endif // SBP_UTILS_HPP