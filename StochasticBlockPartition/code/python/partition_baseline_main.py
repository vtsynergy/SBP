"""Runs the partitioning script.
"""

import timeit
import argparse

from evaluate import evaluate_partition, evaluate_subgraph_partition
from graph import Graph
from samplestack import SampleStack
from sbp import stochastic_block_partition


def parse_arguments():
    """Parses command-line arguments.

        Returns
        -------
        args : argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parts", type=int, default=0, 
                        help="""The number of streaming partitions to the dataset. If the dataset is static, 0.
                             Default = 0""")
    parser.add_argument("-o", "--overlap", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-s", "--blockSizeVar", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-t", "--type", type=str, default="static", 
                        help="(static|streamingEdge|streamingSnowball). Default = static")
    parser.add_argument("-n", "--numNodes", type=int, default=1000, help="The size of the dataset. Default = 1000")
    parser.add_argument("-d", "--directory", type=str, default="../../data", 
                        help="The location of the dataset directory. Default = ../../data")
    parser.add_argument("-v", "--verbose", action="store_true", help="If supplied, will print 'helpful' messages.")
    parser.add_argument("-b", "--blockProposals", type=int, default=10, 
                        help="The number of block merge proposals per block. Default = 10")
    parser.add_argument("-i", "--iterations", type=int, default=100, 
                        help="Maximum number of node reassignment iterations. Default = 100")
    parser.add_argument("-r", "--blockReductionRate", type=float, default=0.5, 
                        help="The block reduction rate. Default = 0.5")
    parser.add_argument("--beta", type=int, default=3, 
                        help="exploitation vs exploration: higher threshold = higher exploration. Default = 3")
    parser.add_argument("--sparse", action="store_true", 
                        help="If supplied, will use Scipy's sparse matrix representation for the matrices.")
    parser.add_argument("-c", "--csv", type=str, default="eval/benchmark", 
                        help="The filepath to the csv file in which to store the evaluation results.")
    # Nodal Update Strategy
    parser.add_argument("-u", "--nodal_update_strategy", type=str, default="original", 
                        help="(original|step|exponential|log). Default = original")
    parser.add_argument("--direction", type=str, default="growth", help="(growth|decay) Default = growth")
    parser.add_argument("-f", "--factor", type=float, default=0.0001, 
                        help="""The factor by which to grow or decay the nodal update threshold. 
                            If the nodal update strategy is step:
                                this value is added to or subtracted from the threshold with every iteration
                            If the nodal update strategy is exponential:
                                this (1 +/- this value) is multiplied by the threshold with every iteration
                            If the nodal update strategy is log:
                                this value is ignored
                            Default = 0.0001""")
    parser.add_argument("-e", "--threshold", type=float, default=5e-4, 
                        help="The threshold at which to stop nodal block reassignment. Default = 5e-4")
    parser.add_argument("-z", "--sample_size", type=int, default=100, 
                        help="The percent of total nodes to use as sample. Default = 100 (no sampling)")
    parser.add_argument("-m", "--sample_type", type=str, default="none", 
                        choices=["uniform_random", "random_walk", "random_jump", "degree_weighted",
                                 "random_node_neighbor", "forest_fire", "expansion_snowball", "none"],
                        help="""Sampling algorithm to use. Default = none""")
    parser.add_argument("--sample_iterations", type=int, default=1,
                        help="The number of sampling iterations to perform. Default = 1")
    parser.add_argument("--degrees", action="store_true", help="Save vertex degrees and exit.")
    parser.add_argument("--notruth", action="store_true", help="No truth partition is available.")
    args = parser.parse_args()
    return args
# End of parse_arguments()


if __name__ == "__main__":
    args = parse_arguments()

    true_partition_available = True
    visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions

    t_start = timeit.default_timer()

    if args.sample_type != "none":
        samplestack = SampleStack(args)
        # graph, vertex_mapping, block_mapping = samplestack.tail()
        # print("Performing stochastic block partitioning on sampled subgraph after {} sampling iterations".format(
        #     args.sample_iterations
        # ))
        # partition, evaluation = stochastic_block_partition(graph, args)
        # print('Combining sampled partition with full graph')
        # subgraph, subgraph_partition, vertex_mapping, block_mapping, evaluation = samplestack.unstack(args)
        subgraph, subgraph_partition, sample, evaluation = samplestack.unstack(args)
        full_graph, full_graph_partition, evaluation = samplestack.extrapolate_sample_partition(
            subgraph_partition, sample.vertex_mapping, args, evaluation
        )
        print(len(samplestack.stack))
    else:
        graph = Graph.load(args)
        t_load = timeit.default_timer()
        t_sample = timeit.default_timer()
        print("Performing stochastic block partitioning")
        # begin partitioning by finding the best partition with the optimal number of blocks
        partition, evaluation = stochastic_block_partition(graph, args)

    t_end = timeit.default_timer()
    print('\nGraph partition took {} seconds'.format(t_end - t_start))

    evaluation.total_runtime(t_start, t_end)

    if args.sample_type != "none":
        evaluation.evaluate_subgraph_sampling(full_graph, subgraph, full_graph_partition, subgraph_partition, sample)
        evaluation.num_nodes = full_graph.num_nodes
        evaluation.num_edges = full_graph.num_edges

        # evaluate output partition against the true partition
        evaluate_subgraph_partition(subgraph.true_block_assignment, subgraph_partition.block_assignment, evaluation)
        evaluate_partition(full_graph.true_block_assignment, full_graph_partition.block_assignment, evaluation)
    else:
        evaluate_partition(graph.true_block_assignment, partition.block_assignment, evaluation)
