
#include <iostream>
#include <string>

#include "argparse/argparse.hpp"

#include "graph.hpp"

/// Parses the command-line arguments passed in by the user.
/// Exits with an error message if parsing is unsuccessful.
argparse::ArgumentParser parse_args(int argc, char* argv[]) {
    argparse::ArgumentParser parser("sbp", "0.0.1");
    parser.add_argument("-o", "--overlap")
        .help("The degree of overlap between communities (low|high|unk)")
        .default_value(std::string("low"))
        .action([](const std::string& value) {
            static const std::vector<std::string> choices = { "low", "high", "unk" };
            if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
                return value;
            }
            return std::string{ "low" };
        });;
    parser.add_argument("-s", "--blocksizevar")
        .help("The variation between the sizes of communities (low|high|unk)")
        .default_value(std::string("low"))
        .action([](const std::string& value) {
            static const std::vector<std::string> choices = { "low", "high", "unk" };
            if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
                return value;
            }
            return std::string{ "low" };
        });;
    parser.add_argument("-t", "--type")
        .help("The type of streaming/name of the graph")
        .default_value(std::string("static"));
    parser.add_argument("-n", "--numvertices")
        .help("The number of vertices in the graph")
        .default_value(std::string("1000"));
    // TODO: simplify the directory structure
    parser.add_argument("-d", "--directory")
        .help("The directory in which the graph is stored. The following structure is assumed:\n"
              "filename for graph: <type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_nodes.tsv\n"
              "filename for truth: <type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_truePartition.tsv\n"
              "directory structure: <directory>/<type>/<overlap>Overlap_<blocksizevar>BlockSizeVar/<filename>\n")
        .default_value(std::string("./data/"));
    parser.add_argument("-c", "--csv")
        .help("The path to the csv file in which the results will be stored, without the suffix. E.g.:\n"
              "if --csv=eval/test, results will be stored in eval/test.csv, and details will be stored\n"
              "in eval/test_details.csv")
        .default_value(std::string("./eval/test"));
    parser.add_argument("--tag")
        .help("The tag value for this run, for differentiating different runs in the result csv file or adding\n"
              "additional parameters to the save file")
        .default_value(std::string("default tag"));
    parser.add_argument("--delimiter")
        .help("The delimiter used in the file storing the graph")
        .default_value(std::string("\t"));
    parser.add_argument("--undirected")
        .help("If used, graph will be treated as undirected")
        .default_value(false)
        .implicit_value(true);
    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cout << parser;
        exit(-1);
    }
    return parser;
}


int main(int argc, char* argv[]) {
    argparse::ArgumentParser args = parse_args(argc, argv);
    std::cout << "Parsed out the arguments" << std::endl;
    // std::cout << args << std::endl;
    Graph graph = Graph::load(args);
}
