/// A wrapper around tclap to make the use of command-line arguments bearable.
#ifndef TCLAP_WRAPPER_ARGS
#define TCLAP_WRAPPER_ARGS

#include <iostream>
#include <omp.h>
#include <string>

#include "tclap/CmdLine.h"

// TODO: add a verbose argument
class Args {
    public:  // Everything in here is public, because why not?
    /** Define all the arguments here for easy access */
    std::string algorithm;
    bool approximate;
    int batches;
    std::string blocksizevar;
    std::string csv;  // TODO: save results in a csv file
    std::string delimiter;
    bool deterministic;
    std::string directory;
    int nodes;  // TODO: remove after this experiment
    int numvertices;
    std::string overlap;
    std::string partition;
    bool sequential;
    std::string tag;  // TODO: add tag to saved results
    int threads;
    std::string type;
    bool undirected;

    /// Parses the command-line options and stores the results in an easy-to-retrieve manner.
    Args(int argc, char** argv) {
        /** Use tclap to retrieve the arguments here */
        try {
            TCLAP::CmdLine parser("Stochastic block blockmodeling algorithm", ' ', "alpha.0.1");
            TCLAP::ValueArg<std::string> algorithm("a", "algorithm", "The algorithm to use for the finetuning/MCMC "
                                                   "step of stochastic block blockmodeling. Note: there is currently no "
                                                   "parallel implementation of metropolis hastings", false,
                                                   "async_gibbs", "async_gibbs|metropolis_hastings", parser);
            TCLAP::SwitchArg approximate("", "approximate", "If set, an approximate version of the block merge "
                                         "step will be used. It's slightly faster, but less accurate for complex "
                                         "graphs.", parser, false);
            TCLAP::ValueArg<int> batches("", "batches", "The number of batches to use for the asynchronous_gibbs "
                                         "algorithm. Too many batches will lead to many updates and little parallelism,"
                                         " but too few will lead to poor results or more iterations", false, 10, "int",
                                         parser);
            TCLAP::ValueArg<std::string> blocksizevar("b", "blocksizevar", "The variation between the sizes of "
                                                      "communities", false, "low", "low|high|unk", parser);
            TCLAP::ValueArg<std::string> csv("c", "csv",
                            "The path to the csv file in which the results will be stored, without the suffix, e.g.:\n"
                            "if --csv=eval/test, results will be stored in eval/test.csv.",
                                             false, "./eval/test", "path", parser);
            TCLAP::ValueArg<std::string> delimiter("", "delimiter", "The delimiter used in the file storing the graph",
                                                   false, "\t", "string, usually `\\t` or `,`", parser);
            TCLAP::SwitchArg deterministic("", "deterministic", "If set, the graph vertices will be traversed in a "
                                "deterministic order. This is likely to reduce accuracy. If sequential is false, this "
                                "option is ignored.", parser, false);
            TCLAP::ValueArg<std::string> directory("d", "directory", 
                                "The directory in which the graph is stored. The following structure is assumed:\n"
                                "filename for graph:"
                                "<type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_nodes.tsv\n"
                                "filename for truth:"
                                "<type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_trueBlockmodel.tsv\n"
                                "directory structure:"
                                "<directory>/<type>/<overlap>Overlap_<blocksizevar>BlockSizeVar/<filename>\n",
                                                   false, "./data", "path", parser);
            TCLAP::ValueArg<int> nodes("", "nodes", "The number of partitions to create", false, 1, "int", parser);
            TCLAP::ValueArg<int> numvertices("n", "numvertices", "The number of vertices in the graph", false, 1000,
                                             "int", parser);
            TCLAP::ValueArg<std::string> overlap("o", "overlap", "The degree of overlap between communities", false,
                                                 "low", "low|high|unk", parser);
            TCLAP::ValueArg<std::string> partition("p", "partition", "The type of partitioning to use to divide the "
                                                   "graph amongst the MPI Processes. Only matters when nprocs > 1",
                                                   false, "round_robin", "round_robin|random|snowball", parser);
            TCLAP::SwitchArg sequential("", "sequential", "If set, all graph vertices will be guaranteed to be "
                                "traversed in each iteration. If this option is not provided (set to false), the "
                                "deterministic option is ignored.", parser, false);
            TCLAP::ValueArg<std::string> tag("", "tag", "The tag value for this run, for differentiating different "
                                             "runs or adding custom parameters to the save file", false, "default tag",
                                             "string or param1=value1;param2=value2", parser);
            TCLAP::ValueArg<int> threads("", "threads", "The number of OpenMP threads to use. If less than 1, will set "
                                         "number of threads to number of logical CPU cores", false, 0, "int", parser);
            TCLAP::ValueArg<std::string> type("t", "type", "The type of streaming/name of the graph", false, "static",
                                              "string", parser);
            TCLAP::SwitchArg undirected("", "undirected", "If set, graph will be treated as undirected", parser,
                                        false);
            parser.parse(argc, argv);
            this->algorithm = algorithm.getValue();
            this->approximate = approximate.getValue();
            this->batches = batches.getValue();
            this->blocksizevar = blocksizevar.getValue();
            this->csv = csv.getValue();
            this->delimiter = delimiter.getValue();
            this->deterministic = deterministic.getValue();
            this->directory = directory.getValue();
            this->nodes = nodes.getValue();
            this->numvertices = numvertices.getValue();
            this->overlap = overlap.getValue();
            this->partition = partition.getValue();
            this->sequential = sequential.getValue();
            this->tag = tag.getValue();
            this->threads = threads.getValue();
            this->type = type.getValue();
            this->undirected = undirected.getValue();
        } catch (TCLAP::ArgException &exception) {
            std::cerr << "ERROR: " << exception.error() << " for argument " << exception.argId() << std::endl;
            exit(-1);
        }
    }
};

#endif // TCLAP_WRAPPER_ARGS
