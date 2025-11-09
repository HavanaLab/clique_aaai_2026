import json
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils.undirected import to_undirected
import torch_geometric.transforms as T
from torch_sparse import SparseTensor


def remove_isolated_nodes(adj_matrix):
    """
    Removes isolated nodes from an adjacency matrix.

    Parameters:
    - adj_matrix: A 2D NumPy array representing the adjacency matrix.

    Returns:
    - A new adjacency matrix with isolated nodes removed.
    """
    # Find indices of rows/columns that are not all zeros
    row_indices = np.any(adj_matrix, axis=1)
    col_indices = np.any(adj_matrix, axis=0)

    # Apply the indices to filter out isolated nodes
    filtered_matrix = adj_matrix[row_indices][:, col_indices]
    return filtered_matrix


def adj_matrix_to_data(adj_matrix):
    #check if adj_matrix contains non 0 or 1 values
    if np.any((adj_matrix != 0) & (adj_matrix != 1)):
        return adj_matrix_to_data_sparse_pca(adj_matrix)
    else:
        return adj_matrix_to_data_clique(adj_matrix)


def adj_matrix_to_data_clique(adj_matrix):
    x = torch.ones((len(adj_matrix)), dtype=torch.float32)
    adj = torch.tensor(adj_matrix, dtype=torch.float32)
    num_nodes = adj.size(0)
    edge_index, _ = dense_to_sparse(adj)
    edge_index = to_undirected(edge_index)
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=None).t()
    data_obj = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, adj_t=adj_t)
    return data_obj



def adj_matrix_to_data_sparse_pca(adj_matrix):
    x = torch.ones((len(adj_matrix)), dtype=torch.float32)
    adj = torch.tensor(adj_matrix, dtype=torch.float32)
    num_nodes = adj.size(0)
    edge_index, edge_attr = dense_to_sparse(adj)
    edge_index = to_undirected(edge_index)
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=None).t()
    data_obj = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, adj_t=adj_t,edge_attr=edge_attr)
    return data_obj

class JSONLDataset(Dataset):
    def __init__(
        self, transform=None, pre_transform=None, file_path=None, max_rows=1000
    ):
        self.file_path = file_path
        super(JSONLDataset, self).__init__(None, transform, pre_transform)
        self.data_list = self.load_data_from_jsonl(max_rows=max_rows)

    @property
    def raw_file_names(self):
        return [self.file_path]

    @property
    def processed_file_names(self):
        return []  # Adjust the filename as needed

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def load_data_from_jsonl(self, max_rows=1000):
        data_list = []
        with open(self.file_path, "r") as f:
            # line_count = 0
            for line_number, line in enumerate(f):
                print("on line", line_number)
                if line_number > max_rows:
                    break
                obj = json.loads(line)
                adj_matrix = obj["adjacency_matrix"]
                data_obj = adj_matrix_to_data(adj_matrix)
                data_list.append(data_obj)
        return data_list


