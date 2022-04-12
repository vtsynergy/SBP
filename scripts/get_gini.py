# Gotten from
from pathlib import Path
import sys

from gini import gini
import graph_tool as gt
import numpy as np

if __name__ == "__main__":
    folder = sys.argv[1]
    paths = Path(folder).rglob("*_nodes.tsv")
    for path in paths:
        print(path)
        graph = gt.load_graph_from_csv(str(path), directed=True, csv_options={'delimiter': '\t'})
        print("V: {0} E: {1}".format(graph.num_vertices(), graph.num_edges()))
        degrees = graph.get_total_degrees(np.arange(graph.num_vertices()))
        # print(type(degrees))
        gini_index = gini.gini(degrees.astype(np.float64))
        print("GINI index: ", gini_index)
