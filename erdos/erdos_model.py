from typing import Literal, Union

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
from torch_geometric.nn import GINConv
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from torch_scatter import scatter_add
from torch_geometric.data import DataLoader
from scipy.sparse import csr_matrix

from erdos.jsonl_dataset import JSONLDataset, adj_matrix_to_data
from graphs.graph_utils_numpy import least_probable_removal as least_probable_removal_numpy



def conv_block(in_f, out_f, momentum, *args, **kwargs):
    return GINConv(
        Sequential(
            Linear(in_f, out_f),
            ReLU(),
            Linear(out_f, out_f),
            ReLU(),
            BatchNorm1d(out_f, momentum=momentum),
            nn.LeakyReLU(),
            GraphSizeNorm(),
            BatchNorm1d(out_f, momentum=momentum),
        ),
        train_eps=True,
    )


class clique_MPNN(torch.nn.Module):
    def __init__(self, num_layers, hidden1, hidden2, deltas, repeated=1):
        super(clique_MPNN, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.momentum = 0.1
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        # Note: the value initially used is 4.
        self.num_layers = num_layers
        # Note: this mechanism(repeated) does not appear in the original work
        # to disable it is equivalent to is being equal to 1
        self.repeated = repeated
        self.heads = 8

        self.convs_list = torch.nn.ModuleList()
        # TODO num layers should come also be saved to pth file
        for i in range(num_layers - 1):
            self.convs_list.append(
                conv_block(
                    self.heads * self.hidden1,
                    self.heads * self.hidden1,
                    momentum=self.momentum,
                )
            )
        self.conv1 = conv_block(self.hidden2, self.heads * self.hidden1, self.momentum)

        self.lin1 = Linear(self.heads * self.hidden1, self.hidden1)
        self.lin2 = Linear(self.hidden1, 1)
        self.gnorm = GraphSizeNorm()

    def reset_parameters(self):

        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def weighted_gin_forward(
        self, old_conv, adj: torch.sparse.Tensor, x: torch.Tensor, edge_attr
    ):
        """
        old_conv: trained GINConv
        adj: sparse adjacency matrix, shape (N, N), values = u (weights)
        x: node features, shape (N, F)
        """
        eps = old_conv.eps
        mlp = old_conv.nn

        num_nodes = x.size(0)

        adj_weighted = adj
        if edge_attr is not None:
            adj_weighted = adj.set_value(edge_attr, layout="coo")

        # Sparse matrix multiplication: weighted neighbor aggregation
        weighted_neighbors = adj_weighted @ x  # (N, F)

        # Add (1+eps) * self-feature
        agg = (1 + eps) * x + weighted_neighbors

        # Pass through MLP
        out = mlp(agg)

        return out

    def forward(self, x, adj_t, return_hidden=False, edge_attr=None):
        hidden = []
        x = x.unsqueeze(-1)
        if edge_attr is not None:
            x = self.weighted_gin_forward(self.conv1, adj_t, x, edge_attr)
        else:
            x = self.conv1(x, adj_t)
        hidden.append(x)
        for _ in range(self.repeated):
            for conv in self.convs_list:
                if edge_attr is not None:
                    x = self.weighted_gin_forward(conv, adj_t, x, edge_attr)
                else:
                    x = conv(x, adj_t)
                hidden.append(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        maxval = torch.max(x)
        minval = torch.min(x)
        if minval != maxval:
            x = (x - minval) / (maxval + 1e-6 - minval)
        if return_hidden:
            return x, hidden
        return x

    def compute_loss(self, prob, adj_t, edge_attr=None, penalty_coefficient=0.25):
        pairwise_prodsums = (prob @ prob.T).sum(dim=1) / 2
        self_sums = (prob * prob).sum(dim=1)

        row, col, _ = adj_t.t().coo()
        # When refactored merged 2 versions 1 uses the edge_attr and the other is the original
        if edge_attr is not None:
            if edge_attr is None:
                # fallback to 1 if no weights
                edge_attr = torch.ones_like(row, dtype=prob.dtype, device=prob.device)

            edge_weights_pred = prob[row] * prob[col]
            weighted_edge_contrib = edge_weights_pred * edge_attr.unsqueeze(-1)

            expected_weight_G = (
                scatter_add(weighted_edge_contrib.unsqueeze(-1), row.unsqueeze(-1), 0) / 2.0
            )

            expected_weight_G = expected_weight_G.squeeze()
            expected_clique_weight = pairwise_prodsums - self_sums
            expected_distance = expected_clique_weight - expected_weight_G
            expected_loss = 0.5 * (
                penalty_coefficient * expected_distance - expected_weight_G
            )
            loss = expected_loss.mean().squeeze()
            print(f"Loss: {loss}")
            return loss
        else:
            edge_index = torch.stack([row, col], dim=0)
            expected_weight_G = (
                scatter_add(
                    prob[row].unsqueeze(-1) * prob[col].unsqueeze(-1), row.unsqueeze(-1), 0
                )
                / 2.0
            )
            expected_weight_G = expected_weight_G.squeeze()
            expected_clique_weight = pairwise_prodsums - self_sums
            expected_distance = expected_clique_weight - expected_weight_G
            expected_loss = 0.5 * (
                penalty_coefficient * expected_distance - expected_weight_G
            )
            loss = expected_loss.mean().squeeze()
            print(f"Loss: {loss}")
            return loss


def train(
    dataset,
    model,
    optimizer,
    epochs=100,
    batch_size=32,
    checkpoint="model.pth",
    device="cuda",
):
    torch.manual_seed(1)
    np.random.seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_trainpoints = int(np.floor(0.6 * len(dataset)))
    num_valpoints = int(np.floor(num_trainpoints / 3))
    num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)

    traindata = dataset[0:num_trainpoints]
    valdata = dataset[num_trainpoints : num_trainpoints + num_valpoints]
    testdata = dataset[num_trainpoints + num_valpoints :]
    train_loader = DataLoader(traindata, batch_size, shuffle=True)
    test_loader = DataLoader(testdata, batch_size, shuffle=False)
    val_loader = DataLoader(valdata, batch_size, shuffle=False)

    for epoch in range(epochs):
        totalretdict = {}
        count = 0

        print("Epoch: ", epoch)
        model.train()
        for data in train_loader:
            count += 1
            optimizer.zero_grad(),
            data = data.to(device)

            probs = model(data.x, data.adj_t, edge_attr=data.edge_attr)
            loss = model.compute_loss(
                probs, data.adj_t, edge_attr=data.edge_attr, penalty_coefficient=4.0
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_layers": model.num_layers,
                "repeated": model.repeated,
            },
            checkpoint,
        )
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_layers": model.num_layers,
                    "repeated": model.repeated,
                },
                f"{checkpoint}-{epoch}",
            )


class ErdosModel:
    def __init__(
        self,
        pth_file: str,
        device: str = "cuda",
        model_state_dict_key="model_state_dict",
        num_layers=4,
        repeated=1,
    ):

        checkpoint = torch.load(pth_file, map_location=torch.device(device))
        if isinstance(checkpoint, dict) and "num_layers" in checkpoint.keys():
            num_layers = checkpoint["num_layers"]
        if isinstance(checkpoint, dict) and "repeated" in checkpoint.keys():
            repeated = checkpoint["repeated"]
        if model_state_dict_key:
            checkpoint = checkpoint[model_state_dict_key]
        self.model = clique_MPNN(num_layers, 64, 1, 1, repeated=repeated)
        self.device = device
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.to(self.device)

    def compute_prob(self, adj_matrix, sparse=True, return_hidden=False, *args, **kwargs):
        data = adj_matrix_to_data(adj_matrix)
        data = data.to(self.device)
        if sparse:
            probs = self.model(data.x, data.adj_t, return_hidden=return_hidden, edge_attr=data.edge_attr, *args, **kwargs)
        else:
            probs = self.model(data.x, data.edge_index, return_hidden=return_hidden, *args, **kwargs)
        return probs

    def least_probable_removal(
        self,
        adj_matrix,
        compute_once=False,
        reversed_order=False,
        should_remove_isolated_nodes=False,
        use_1st_pc=False,
        use_1st_pc_layer=None,
        fix_by_spearman=False,
        should_split_to_cc: Union[Literal["Each"] | Literal["Once"] | Literal["None"]] = "None",
        *args, **kwargs

    ):
        if type(adj_matrix) == tuple:
            probs = self.compute_prob(adj_matrix[1], *args, **kwargs)
        else:
            probs = self.compute_prob(adj_matrix, *args, **kwargs)
        probs = probs.squeeze()

        def compute_prob(H, return_hidden=False):
            probs, hidden = self.compute_prob(H, return_hidden=True)
            if use_1st_pc and use_1st_pc_layer is not None:
                embedding = hidden[use_1st_pc_layer].detach().numpy()
                pca = PCA(n_components=1)
                pca.fit(embedding)
                pc1 = (embedding - pca.mean_) @ (pca.components_.T)
                probs = pc1
            else:
                probs = probs.detach().cpu().numpy().flatten()
            if reversed_order:
                probs = -1 * probs
            if return_hidden is False:
                return probs
            else:
                return probs, [h.tolist() for h in hidden]

        return least_probable_removal_numpy(
            adj_matrix,
            compute_prob,
            compute_once=compute_once,
            should_remove_isolated_nodes=should_remove_isolated_nodes,
            fix_by_spearman=fix_by_spearman,
            *args, **kwargs
        )
