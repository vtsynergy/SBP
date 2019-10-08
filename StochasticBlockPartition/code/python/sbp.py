"""The stochastic block partitioning algorithm.
"""

import argparse
import timeit
from typing import Tuple

from block_merge import merge_blocks
from evaluation import Evaluation
from graph import Graph
from node_reassignment import reassign_nodes, propagate_membership, fine_tune_membership
from partition import Partition, PartitionTriplet
from partition_baseline_support import plot_graph_with_partition
from partition_baseline_support import prepare_for_partition_on_next_num_blocks


def stochastic_block_partition(graph: Graph, args: argparse.Namespace,
    init_partition: Partition = None, init_evaluation: Evaluation = None) -> Tuple[Partition, Evaluation]:
    """The stochastic block partitioning algorithm
    """ 
    visualize_graph = False

    if init_partition is None and init_evaluation is None:
        partition = Partition(graph.num_nodes, graph.out_neighbors, args)
        evaluation = Evaluation(args, graph)
    else:
        partition = init_partition
        evaluation = init_evaluation
    # print(partition.interblock_edge_count._matrix)

    # initialize items before iterations to find the partition with the optimal number of blocks
    partition_triplet = PartitionTriplet()
    graph_object = None
    
    while not partition_triplet.optimal_num_blocks_found:
        ##################
        # BLOCK MERGING
        ##################
        # begin agglomerative partition updates (i.e. block merging)
        t_block_merge_start = timeit.default_timer()

        if args.verbose:
            print("\nMerging down blocks from {} to {}".format(partition.num_blocks,
                                                               partition.num_blocks - partition.num_blocks_to_merge))
        
        partition = merge_blocks(partition, args.blockProposals, args.sparse, graph.out_neighbors, evaluation)

        t_nodal_update_start = timeit.default_timer()

        ############################
        # NODAL BLOCK UPDATES
        ############################
        if args.verbose:
            print("Beginning nodal updates")

        partition = reassign_nodes(partition, graph, partition_triplet, evaluation, args)

        if visualize_graph:
            graph_object = plot_graph_with_partition(graph.out_neighbors, partition.block_assignment, graph_object)

        t_prepare_next_start = timeit.default_timer()

        # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try
        partition, partition_triplet = prepare_for_partition_on_next_num_blocks(
            partition, partition_triplet, args.blockReductionRate)

        t_prepare_next_end = timeit.default_timer()
        evaluation.update_timings(t_block_merge_start, t_nodal_update_start, t_prepare_next_start, t_prepare_next_end)
        evaluation.num_iterations += 1

        if args.verbose:
            print('Overall entropy: {}'.format(partition_triplet.overall_entropy))
            print('Number of blocks: {}'.format(partition_triplet.num_blocks))
            if partition_triplet.optimal_num_blocks_found:
                print('\nOptimal partition found with {} blocks'.format(partition.num_blocks))
    return partition, evaluation
# End of stochastic_block_partition()