from pmc.pmc import pmc


def get_max_clique_size(G):
    max_clique = get_max_clique(G)
    max_clique_size = len(max_clique)
    return max_clique_size


def get_max_clique(G, to_root=""):
    directed_graph = G.to_directed()
    ei = []
    ej = []
    for edge in directed_graph.edges:
        ei.append(edge[0])
        ej.append(edge[1])
    number_of_nodes = len(directed_graph.nodes)
    number_of_edges = len(ei)
    max_clique = pmc(ei, ej, number_of_nodes, number_of_edges, to_root)
    return max_clique
