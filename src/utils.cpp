#include "utils.hpp"

#include <nlohmann/json.hpp>

#include "mpi_data.hpp"

namespace utils {

std::vector<long> argsort(const std::vector<long> &v) {
    if (v.empty()) {
        return {};
    }

    // isolate integer byte by index.
    auto bmask = [](long x, size_t i) {
        return (static_cast<unsigned long>(x) >> i*8) & 0xFF;
    };

    // allocate temporary buffer.
    std::vector<long> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<long> new_indices(v.size());
    std::vector<long> v_copy(v);
    std::vector<long> new_v(v.size());

    // for each byte in integer (assuming 4-byte int).
    for (size_t i, j = 0; j < sizeof(long); j++) {
        // initialize counter to zero;
        size_t h[256] = {}, start;

        // histogram.
        // count each occurrence of indexed-byte value.
        for (i = 0; i < v_copy.size(); i++)
            h[255 - bmask(v_copy[i], j)]++;

        // accumulate.
        // generate positional offsets. adjust starting point
        // if most significant digit.
        start = (j != 7) ? 0 : 128;
        for ( i = 1+start; i < 256+start; i++ )
            h[i % 256] += h[(i-1) % 256];

        // distribute.
        // stable reordering of elements. backward to avoid shifting
        // the counter array.
        for ( i = v.size(); i > 0; i-- ) {
            size_t k = --h[255 - bmask(v_copy[i - 1], j)];
            new_indices[k] = indices[i - 1];
            new_v[k] = v_copy[i - 1];
//            new_indices[--h[255 - bmask(v[i-1], j)]] = indices[i-1];
        }

        std::swap(indices, new_indices);
        std::swap(v_copy, new_v);
    }
    return indices;
}

std::string build_filepath() {
//    std::ostringstream filepath_stream;
//    filepath_stream << args.directory << "/" << args.type << "/" << args.overlap << "Overlap_" << args.blocksizevar;
//    filepath_stream << "BlockSizeVar/" << args.type << "_" << args.overlap << "Overlap_" << args.blocksizevar;
//    filepath_stream << "BlockSizeVar_" << args.numvertices << "_nodes";
//    // TODO: Add capability to process multiple "streaming" graph parts
//    std::string filepath = filepath_stream.str();
    std::string filepath = args.filepath;
    if (!fs::exists(filepath + ".tsv") && !fs::exists(filepath + ".mtx")) {
        std::cerr << "ERROR " << "File doesn't exist: " << filepath + ".tsv/.mtx" << std::endl;
        exit(-1);
    }
    return filepath;
}

std::vector<std::vector<std::string>> read_csv(fs::path &filepath) {
    std::vector<std::vector<std::string>> contents;
    if (!fs::exists(filepath)) {
        if (mpi.rank == 0)
            std::cerr << "ERROR " << "File doesn't exist: " << filepath << std::endl;
        return contents;
    }
    std::string line;
    std::ifstream file(filepath);
    long items = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream line_stream(line);
        std::string value;
        while (line_stream >> value) {
            row.push_back(value);
            items++;
        }
        contents.push_back(row);
    }
    if (mpi.rank == 0)
        std::cout << "Read in " << contents.size() << " lines and " << items << " values." << std::endl;
    return contents;
}

void insert(NeighborList &neighbors, long from, long to) {
    if (from >= (long) neighbors.size()) {
        std::vector<std::vector<long>> padding(from - neighbors.size() + 1, std::vector<long>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    neighbors[from].push_back(to);
}

void insert_nodup(NeighborList &neighbors, long from, long to) {
    if (from >= (long) neighbors.size()) {
        std::vector<std::vector<long>> padding(from - neighbors.size() + 1, std::vector<long>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    for (const long &neighbor: neighbors[from])
        if (neighbor == to) return;
    neighbors[from].push_back(to);
}

bool insert(std::unordered_map<long, long> &map, long key, long value) {
    std::pair<std::unordered_map<long, long>::iterator, bool> result = map.insert(std::make_pair(key, value));
    return result.second;
}

void radix_sort(std::vector<long> &v) {
    if (v.empty()) {
        return;
    }

    // isolate integer byte by index.
    auto bmask = [](long x, size_t i) {
        return (static_cast<unsigned long>(x) >> i*8) & 0xFF;
    };

    // allocate temporary buffer.
    std::vector<long> sorted(v.size());

    // for each byte in integer (assuming 4-byte int).
    for (size_t i, j = 0; j < sizeof(long); j++) {
        // initialize counter to zero;
        size_t h[256] = {}, start;

        // histogram.
        // count each occurrence of indexed-byte value.
        for (i = 0; i < v.size(); i++)
            h[255 - bmask(v[i], j)]++;

        // accumulate.
        // generate positional offsets. adjust starting point
        // if most significant digit.
        start = (j != 7) ? 0 : 128;
        for ( i = 1+start; i < 256+start; i++ )
            h[i % 256] += h[(i-1) % 256];

        // distribute.
        // stable reordering of elements. backward to avoid shifting
        // the counter array.
        for ( i = v.size(); i > 0; i-- ) {
            sorted[--h[255 - bmask(v[i-1], j)]] = v[i-1];
        }

        std::swap(v, sorted);
    }
}

void radix_sort(std::vector<std::pair<long, long>> &v) {
    if (v.empty()) {
        return;
    }

    // isolate integer byte by index.
    auto bmask = [](long x, size_t i) {
        return (static_cast<unsigned long>(x) >> i*8) & 0xFF;
    };

    // allocate temporary buffer.
    std::vector<std::pair<long, long>> sorted(v.size());

    // for each byte in integer (assuming 4-byte int).
    for (size_t i, j = 0; j < sizeof(long); j++) {
        // initialize counter to zero;
        size_t h[256] = {}, start;

        // histogram.
        // count each occurrence of indexed-byte value.
        for (i = 0; i < v.size(); i++)
            h[255 - bmask(v[i].second, j)]++;

        // accumulate.
        // generate positional offsets. adjust starting point
        // if most significant digit.
        start = (j != 7) ? 0 : 128;
        for ( i = 1+start; i < 256+start; i++ )
            h[i % 256] += h[(i-1) % 256];

        // distribute.
        // stable reordering of elements. backward to avoid shifting
        // the counter array.
        for ( i = v.size(); i > 0; i-- ) {
            sorted[--h[255 - bmask(v[i-1].second, j)]] = v[i-1];
        }

        std::swap(v, sorted);
    }
}

void write_json(const std::vector<long> &block_assignment, double description_length, ulong MCMC_moves,
                ulong MCMC_iterations, double runtime) {
    nlohmann::json output;
    output["Runtime (s)"] = runtime;
    output["Filepath"] = args.filepath;
    output["Tag"] = args.tag;
    output["Algorithm"] = args.algorithm;
    output["Degree Product Sort"] = args.degreeproductsort;
    output["Data Distribution"] = args.distribute;
    output["Greedy"] = args.greedy;
    output["Metropolis-Hastings Ratio"] = args.mh_percent;
    output["Overlap"] = args.overlap;
    output["Block Size Variation"] = args.blocksizevar;
    output["Sample Size"] = args.samplesize;
    output["Sampling Algorithm"] = args.samplingalg;
    output["Num. Subgraphs"] = args.subgraphs;
    output["Subgraph Partition"] = args.subgraphpartition;
    output["Num. Threads"] = args.threads;
    output["Num. Processes"] = mpi.num_processes;
    output["Type"] = args.type;
    output["Undirected"] = args.undirected;
    output["Num. Vertex Moves"] = MCMC_moves;
    output["Num. MCMC Iterations"] = MCMC_iterations;
    output["Results"] = block_assignment;
    output["Description Length"] = description_length;
    fs::create_directories(fs::path(args.json));
    std::ostringstream output_filepath_stream;
    output_filepath_stream << args.json << "/" << args.output_file;
    std::string output_filepath = output_filepath_stream.str();
    std::cout << "Saving results to file: " << output_filepath << std::endl;
    std::ofstream output_file;
    output_file.open(output_filepath, std::ios_base::app);
    output_file << std::setw(4) << output << std::endl;
    output_file.close();
}

}  // namespace utils
