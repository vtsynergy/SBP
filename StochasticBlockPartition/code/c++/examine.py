"""Examine a graph
"""
from collections import namedtuple
import csv
from util import _load_graph

from graph_tool import Graph
from graph_tool import load_graph_from_csv
from graph_tool.clustering import global_clustering
from graph_tool.stats import vertex_average
from graph_tool.topology import extract_largest_component
import numpy as np


Params = namedtuple("Params", ["communities", "mindeg", "maxdeg", "exp", "overlap", "sizevar", "vertices"])
Fieldnames = [
    "experiment", "name", "real", "vertices", "edges", "min_degree", "max_degree", "avg_degree",
    "largest_connected_component", "islands", "clustering_coefficient"
]
Properties = namedtuple("Properties", Fieldnames)

REAL = [
    "web-polblogs", "web-google", "web-edu", "web-EPA", "web-BerkStan", "web-webbase-2001", "web-indochina-2004",
    "web-spam", "web-sk-2005", "web-frwikinews-user-edits", "web-NotreDame", "web-arabic-2005", "web-italycnr-2000",
    "web-Stanford", "web-baidu-baike-related", "web-it-2005", "web-google-dir", "web-BerkStan-dir",
    "web-wikipedia2009", "web-uk-2005", "web-wiki-ch-internal", "web-hudong", "web-baidu-baike",
    "web-wikipedia-growth", "web-wikipedia-link-de", "web-wikipedia-link-it", "web-wikipedia-link-fr",
    "web-indochina-2004-all", "web-uk-2002-all", "web-sk-2005-all", "web-webbase-2001-all", "web-it-2004-all",
    "web-sk-2005-all"
]

REAL_UNDIRECTED = [
    "web-arabic-2005", "web-BerkStan", "web-edu", "web-google", "web-indochina-2004", "web-it-2004", "web-polblogs",
    "web-sk-2005", "web-spam", "web-uk-2005", "web-webbase-2001", "web-wikipedia2009"
]

GENERATED = {
    "distribution": [
        Params(32, 1, 100, -3.5, 2.0, 5.0, 20000),
        Params(32, 1, 100, -3.0, 2.0, 5.0, 20000),
        Params(32, 1, 100, -2.5, 2.0, 5.0, 20000),
        Params(32, 1, 100, -2.0, 2.0, 5.0, 20000),
        Params(32, 1, 100, -1.5, 2.0, 5.0, 20000),
        Params(32, 1, 100, -1.0, 2.0, 5.0, 20000),
        Params(32, 1, 100, 0, 2.0, 5.0, 20000),
        Params(32, 1, 100, 1.0, 2.0, 5.0, 20000),
        Params(32, 1, 100, 1.5, 2.0, 5.0, 20000),
        Params(32, 1, 100, 2.0, 2.0, 5.0, 20000),
        Params(32, 1, 100, 2.5, 2.0, 5.0, 20000),
        Params(32, 1, 100, 3.0, 2.0, 5.0, 20000),
        Params(32, 1, 100, 3.5, 2.0, 5.0, 20000),
    ],
    "communitysize": [
        Params(2, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(4, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(8, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(16, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(32, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(64, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(128, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(256, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(512, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(1024, 1, 100, -2.1, 2.0, 5.0, 32768),
        Params(2048, 1, 100, -2.1, 2.0, 5.0, 32768),
    ],
    "overlap": [
        Params(32, 1, 100, -2.1, 1.0, 5.0, 20000),
        Params(32, 1, 100, -2.1, 1.5, 5.0, 20000),
        Params(32, 1, 100, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 100, -2.1, 2.5, 5.0, 20000),
        Params(32, 1, 100, -2.1, 3.0, 5.0, 20000),
        Params(32, 1, 100, -2.1, 3.5, 5.0, 20000),
        Params(32, 1, 100, -2.1, 4.0, 5.0, 20000),
        Params(32, 1, 100, -2.1, 4.5, 5.0, 20000),
        Params(32, 1, 100, -2.1, 5.0, 5.0, 20000),
        Params(32, 1, 100, -2.1, 5.5, 5.0, 20000),
        Params(32, 1, 100, -2.1, 6.0, 5.0, 20000),
    ],
    "sizevariation": [
        Params(32, 1, 100, -2.1, 2.0, 1.0, 20000),
        Params(32, 1, 100, -2.1, 2.0, 1.5, 20000),
        Params(32, 1, 100, -2.1, 2.0, 2.0, 20000),
        Params(32, 1, 100, -2.1, 2.0, 2.5, 20000),
        Params(32, 1, 100, -2.1, 2.0, 3.0, 20000),
        Params(32, 1, 100, -2.1, 2.0, 3.5, 20000),
        Params(32, 1, 100, -2.1, 2.0, 4.0, 20000),
        Params(32, 1, 100, -2.1, 2.0, 4.5, 20000),
        Params(32, 1, 100, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 100, -2.1, 2.0, 5.5, 20000),
        Params(32, 1, 100, -2.1, 2.0, 6.0, 20000),
    ],
    "supernode": [
        Params(32, 1, 16, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 32, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 64, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 128, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 256, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 512, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 1024, -2.1, 2.0, 5.0, 20000),
        Params(32, 1, 2048, -2.1, 2.0, 5.0, 20000),
    ],
    "scaling": [
        Params(11, 1, 16, -2.1, 2.0, 5.0, 1000),
        Params(14, 1, 16, -2.1, 2.0, 5.0, 2000),
        Params(18, 1, 16, -2.1, 2.0, 5.0, 4000),
        Params(23, 1, 16, -2.1, 2.0, 5.0, 8000),
        Params(29, 1, 16, -2.1, 2.0, 5.0, 16000),
        Params(37, 1, 16, -2.1, 2.0, 5.0, 32000),
        Params(48, 1, 16, -2.1, 2.0, 5.0, 64000),
        Params(61, 1, 16, -2.1, 2.0, 5.0, 128000),
        Params(78, 1, 16, -2.1, 2.0, 5.0, 256000),
        Params(99, 1, 16, -2.1, 2.0, 5.0, 512000),
        Params(126, 1, 16, -2.1, 2.0, 5.0, 1024000),
    ],
    "sparsity": [

    ]
}


def examine_graph(graph: Graph, experiment: str, graphname: str, real: bool) -> Properties:
    vertices = graph.num_vertices()
    edges = graph.num_edges()
    total_degrees = graph.get_total_degrees(np.arange(vertices))
    min_degree = np.min(total_degrees)
    max_degree = np.max(total_degrees)
    avg_degree = vertex_average(graph, "total")[0]
    largest_component = extract_largest_component(graph, directed=False).num_vertices()
    num_islands = np.sum(total_degrees == 0)
    cc = global_clustering(graph)[0]
    return Properties(experiment, graphname, real, vertices, edges, min_degree, max_degree, avg_degree,
                      largest_component, num_islands, cc)
# End of examine_graph()


if __name__ == "__main__":
    # Examine real world graphs
    print("=====Examining Real World Graphs=====")
    props = list()
    for graphname in REAL:
        print("graphname = ", graphname)
        filename = "../../data/{0}/unkOverlap_unkBlockSizeVar/{0}_unkOverlap_unkBlockSizeVar_-1_nodes".format(graphname)
        if graphname in REAL_UNDIRECTED:
            graph = load_graph_from_csv(filename + ".tsv", False, csv_options={'delimiter': ' '})
        else:
            graph = load_graph_from_csv(filename + ".tsv", True, csv_options={'delimiter': ' '})
        props.append(examine_graph(graph, "real", graphname, True))
    with open('real_graphs_examined.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=Fieldnames)
        writer.writeheader()
        for prop in props:
            writer.writerow(prop._asdict())
    # Examine synthetic graphs
    print("=====Examining Synthetic Graphs=====")
    props = list()
    for experiment, graph_params in GENERATED.items():
        for param in graph_params:
            graphtype = "test_{0}_{1}_{2}_{3}".format(param.communities, param.mindeg, param.maxdeg, param.exp)
            difficulty = "{0}Overlap_{1}BlockSizeVar".format(param.overlap, param.sizevar)
            print("experiment = {0}, graphname = {1}_{2}".format(experiment, graphtype, difficulty))
            filename = "../../data/{0}/{1}/{0}_{1}_{2}_nodes".format(graphtype, difficulty, param.vertices)
            graph = _load_graph(filename)
            props.append(examine_graph(graph, "experiment", "{}_{}".format(graphtype, difficulty), False))
    with open('synthetic_graphs_examined.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=Fieldnames)
        writer.writeheader()
        for prop in props:
            writer.writerow(prop._asdict())
