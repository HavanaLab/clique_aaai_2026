
import json
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
from scipy.sparse import csr_matrix
import networkx as nx

from graphs.clique_solver import get_max_clique_size


# imdb_binary = TUDataset(root='data/TUDataset', name='IMDB-BINARY')


def generate_rows(dataset):
    for i in range(len(dataset)):
        data = dataset[i]  # Get the first graph object.
        adj = to_dense_adj(data.edge_index).numpy()[0]
        sparse_similarity_matrix = csr_matrix(adj)
        G = nx.from_scipy_sparse_array(sparse_similarity_matrix)
        G = G.to_undirected()
        max_clique_size = get_max_clique_size(G)
        yield {"adjacency_matrix": adj.tolist(), "max_clique_size": max_clique_size, "easy_indicator": 0, "medium_indicator": 0, "hard_indicater": 0}


if __name__ == "__main__":
    collab = TUDataset(root='data/TUDataset', name='COLLAB')
    dataset = collab
    dataset_name = "datasets/collab.jsonl"
    with open(dataset_name, "w") as f:
       for row in generate_rows(dataset):
           json.dump(row, f)
           f.write("\n")