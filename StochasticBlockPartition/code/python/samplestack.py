"""A Stack object for incremental sampling
"""
from typing import Tuple, Dict, List
import timeit

from evaluation import Evaluation
from graph import Graph
from node_reassignment import fine_tune_membership
from partition import Partition
from samplestate import SampleState
from sbp import stochastic_block_partition


class SampleStack(object):
    def __init__(self, args: 'argparse.Namespace') -> None:
        """Creates a SampleStack object.

        Parameters
        ---------
        args : argparse.Namespace
            the command-line arguments provided by the user
        """
        # Load graph
        # Create stack of samples
        # Use List as stack
        self.t_load_start = timeit.default_timer()
        self.full_graph = Graph.load(args)
        self.t_load_end = timeit.default_timer()
        self.stack = list()  # type: List[Tuple[Graph, Dict[int,int], Dict[int,int]]]
        self._sample(args)
        # self.stack.append((graph, None, None))
        self.t_sample_end = timeit.default_timer()
    # End of __init__()

    def _sample(self, args):
        # Iteratively perform sampling
        state = SampleState()
        # graph = self.full_graph
        for _ in range(args.sample_iterations):
            # graph = self.stack[-1][0]
            state, subgraph, vertex_mapping, block_mapping = self.full_graph.sample(args, state)
            self.stack.append((subgraph, vertex_mapping, block_mapping))
    # End of _sample()

    def _push(self):
        # Add a subsample to the stack
        raise NotImplementedError()

    def _pop(self) -> Tuple[Graph, Dict, Dict]:
        # Propagate a subsample's results up the stack
        return self.stack.pop(0)
    # End of _pop()

    def unstack(self, args: 'argparse.Namespace', subgraph_partition: Partition = None,
        evaluation: Evaluation = None) -> Tuple[Graph, Partition, Dict, Dict, Evaluation]:
        """Performs SBP on the first (innermost) sample. Merges said sample with the next in the stack, and performs
        SBP on the combined results. Repeats the process until all samples have been partitioned.

            Paramters
            ---------
            args : argparse.Namespace
                the command-line arguments supplied by the user
            subgraph_partition : Partition
                the current partitioned state of the subgraph. Default = None
            evaluation : Evaluation
                the current state of the evaluation of the algorithm. Default = None

            Returns
            -------
            subgraph : Graph
                the Graph object describing the combined samples
            subgraph_partition : Partition
                the partition results of the combined samples
            vertex_mapping : Dict[int, int]
                the mapping of the vertices from the combined sample to the full graph
            block_mapping : Dict[int, int]
                the mapping of the communities/blocks from the combined sample to the full graph
        """
        # Propagate results back through the stack
        subgraph, vertex_mapping, block_mapping = self._pop()
        subgraph_partition, evaluation = stochastic_block_partition(subgraph, args, subgraph_partition, evaluation)
        while len(self.stack) > 0:
            t1 = timeit.default_timer()
            subgraph, vertex_mapping, block_mapping = self._pop()
            subgraph_partition = Partition.extend_sample(subgraph_partition.num_blocks, subgraph.out_neighbors,
                                                         subgraph_partition.block_assignment, args)
            t2 = timeit.default_timer()
            subgraph_partition, evaluation = stochastic_block_partition(subgraph, args, subgraph_partition, evaluation)
            evaluation.propagate_membership += (t2 - t1)
        return subgraph, subgraph_partition, vertex_mapping, block_mapping, evaluation
    # End of unstack()

    def extrapolate_sample_partition(self, subgraph_partition: Partition, vertex_mapping: Dict[int,int], 
        args: 'argparse.Namespace', evaluation: Evaluation) -> Tuple[Graph, Partition, Evaluation]:
        """Extrapolates the partitioning results from the sample to the full graph.

            This is done by first assigning to every unsampled vertex, the community to which it's most strongly 
            connected. Then, a fine-tuning step (MCMC updates using a modified Metropolis-Hasting algorithm) is run
            on the result.

            Parameters
            ----------
            subgraph_partition : Partition
                the current partitioned state of the subgraph
            evaluation : Evaluation
                the current state of the evaluation of the algorithm
            args : argparse.Namespace
                the command-line arguments supplied by the user
            
            Returns
            -------
            full_graph : Graph
                the graph object representing the entire (unsampled) graph
            full_graph_partition : Partition
                the partition state of the full graph after extrapolation and fine-tuning
            evaluation : Evaluation
                the evaluation results of the algorithm
        """
        t1 = timeit.default_timer()
        full_graph_partition = Partition.from_sample(subgraph_partition.num_blocks, self.full_graph.out_neighbors,
                                                     subgraph_partition.block_assignment, vertex_mapping, args)
        t2 = timeit.default_timer()
        full_graph_partition = fine_tune_membership(full_graph_partition, self.full_graph, evaluation, args)
        t3 = timeit.default_timer()
        evaluation.loading = self.t_load_end - self.t_load_start
        evaluation.sampling = self.t_sample_end - self.t_load_end
        evaluation.propagate_membership += (t2 - t1)
        evaluation.finetune_membership += (t3 - t2)
        return self.full_graph, full_graph_partition, evaluation
    # End of extrapolate_sample_partition()

    def tail(self) -> Tuple[Graph, Partition, Dict]:
        # Get innermost sample
        return self.stack[0]
    # End of tail()
# End of SampleStack()
