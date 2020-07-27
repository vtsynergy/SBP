#include "utils.hpp"

std::string utils::build_filepath(argparse::ArgumentParser &args) {
    std::string directory = args.get("--directory");
    std::string type = args.get("--type");
    std::string overlap = args.get<std::string>("--overlap");
    std::string blocksizevar = args.get<std::string>("--blocksizevar");
    std::string vertices = args.get<std::string>("--numvertices");
    std::ostringstream filepathstream;
    filepathstream << directory << "/" << type << "/" << overlap << "Overlap_" << blocksizevar;
    filepathstream << "BlockSizeVar/" << type << "_" << overlap << "Overlap_" << blocksizevar;
    filepathstream << "BlockSizeVar_" << vertices << "_nodes";
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string filepath = filepathstream.str();
    if (!std::filesystem::exists(filepath + ".tsv")) {
        std::cerr << "File doesn't exist: " << filepath + ".tsv" << std::endl;
        exit(-1);
    }
    return filepath;
}

std::vector<std::vector<std::string>> utils::read_csv(std::filesystem::path &filepath) {
    std::vector<std::vector<std::string>> contents;
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File doesn't exist: " << filepath << std::endl;
        return contents;
    }
    std::string line;
    std::ifstream file(filepath);
    int items = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream linestream(line);
        std::string value;
        while (linestream >> value) {
            row.push_back(value);
            items++;
        }
        contents.push_back(row);
    }
    std::cout << "Read in " << contents.size() << " lines and " << items << " values." << std::endl;
    return contents;
}

void utils::insert(NeighborList &neighbors, int from, int to) {
    if (from >= neighbors.size()) {
        std::vector<std::vector<int>> padding(from - neighbors.size() + 1, std::vector<int>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    neighbors[from].push_back(to);
}
