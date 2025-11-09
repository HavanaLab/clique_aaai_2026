from sklearn.decomposition import PCA
import torch.nn
from torch_geometric.utils import from_networkx
import scipy.sparse as scipy_sparse

from graphs.graph_utils_numpy import (
    least_probable_removal as least_probable_removal_numpy,
)
from torch.nn import Linear
import torch.nn
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data


import torch.nn as nn

from torch_geometric.utils.convert import to_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def numpy_to_edge_index(adj_matrix):
    edge_index = np.argwhere(adj_matrix > 0)
    return edge_index


def numpy_to_pyg_data(adj_matrix):
    edge_index = numpy_to_edge_index(adj_matrix)
    data = Data(
        edge_index=torch.tensor(edge_index).t().contiguous(),
        num_nodes=adj_matrix.shape[0],
    )
    return data


###########


def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = scipy_sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def normalizemx(mx):
    degrees = mx.sum(axis=0)[0].tolist()
    #    print(degrees)
    D = scipy_sparse.diags(degrees, [0])
    D = D.power(-1)
    mx = mx.dot(D)
    return mx


##############################


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_loss(edge_index, output_dis, penalty_coefficient=0.005, device="cuda"):
    return loss(edge_index, output_dis, penalty_coefficient=penalty_coefficient, device=device)

def loss(edge_index, output_dis, penalty_coefficient=0.005, device="cuda"):
    adjmatrix = to_scipy_sparse_matrix(edge_index)
    adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix).to(device)
    I_n = sp.eye(adjmatrix.size(0))
    I_n = sparse_mx_to_torch_sparse_tensor(I_n).cuda()
    Fullm = torch.ones(I_n.size(0), I_n.size(1)).cuda() - I_n  # (N,N)
    diffusionprob = torch.mm(Fullm - adjmatrix, output_dis)
    elewiseloss = output_dis * diffusionprob

    lossComplE = penalty_coefficient * torch.sum(elewiseloss)  # loss on compl of Edges
    lossE = torch.sum(output_dis * torch.mm(adjmatrix, output_dis))
    loss = -lossE + lossComplE
    return loss


# ref: https://discuss.pytorch.org/t/creating-a-sparse-tensor-from-csr-matrix/13658/4
def GCN_diffusion(sptensor, order, feature, device="cuda",should_use_sparse=False):
    """
    Creating a normalized adjacency matrix with self loops.
    sptensor = W
    https://arxiv.org/pdf/1609.02907.pdf
    """
    if should_use_sparse:
        I_n = sp.eye(sptensor.size(0))
        I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
        A_gcn = sptensor + I_n
        degrees = torch.sparse.sum(A_gcn, 0)
        D = degrees
        D = D.to_dense()  # transfer D from sparse tensor to normal torch tensor
        D = torch.pow(D, -0.5)
        D = D.unsqueeze(dim=1)
        gcn_diffusion_list = []
        A_gcn_feature = feature
        for i in range(order):
            A_gcn_feature = torch.mul(A_gcn_feature, D)
            A_gcn_feature = torch.spmm(A_gcn, A_gcn_feature)
            A_gcn_feature = torch.mul(A_gcn_feature, D)
            gcn_diffusion_list += [
                A_gcn_feature,
            ]
        return gcn_diffusion_list
    I_n = torch.eye(sptensor.size(0)).to(device)
    A_gcn = sptensor + I_n
    degrees = torch.sum(A_gcn, 0)
    D = degrees
    D = torch.pow(D, -0.5)
    D = D.unsqueeze(dim=1)
    gcn_diffusion_list = []
    A_gcn_feature = feature
    for i in range(order):
        A_gcn_feature = torch.mul(A_gcn_feature, D)
        A_gcn_feature = torch.mm(A_gcn, A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature, D)
        gcn_diffusion_list += [
            A_gcn_feature,
        ]
    return gcn_diffusion_list

def scattering_diffusion(sptensor, feature, should_use_sparse=False):
    """
    A_tilte,adj_p,shape(N,N)
    feature:shape(N,3) :torch.FloatTensor
    all on cuda
    """

    h_sct1, h_sct2, h_sct3 = SCT1stv2(sptensor, 3, feature, should_use_sparse)

    return h_sct1, h_sct2, h_sct3

def SCT1stv2(sptensor, order, feature,should_use_sparse=False):
    """
    sptensor = W
    """
    if should_use_sparse:
        degrees = torch.sparse.sum(sptensor, 0)
        D = degrees
        D = D.to_dense()  # transfer D from sparse tensor to normal torch tensor
        D = torch.pow(D, -1)
        D = D.unsqueeze(dim=1)
        iteration = 2 ** order
        scale_list = list(2 ** i - 1 for i in range(order + 1))
        feature_p = feature
        sct_diffusion_list = []
        for i in range(iteration):
            D_inv_x = D * feature_p
            W_D_inv_x = torch.spmm(sptensor, D_inv_x)
            feature_p = 0.5 * feature_p + 0.5 * W_D_inv_x
            if i in scale_list:
                sct_diffusion_list += [
                    feature_p,
                ]
        sct_feature1 = sct_diffusion_list[0] - sct_diffusion_list[1]
        sct_feature2 = sct_diffusion_list[1] - sct_diffusion_list[2]
        sct_feature3 = sct_diffusion_list[2] - sct_diffusion_list[3]
        return sct_feature1, sct_feature2, sct_feature3
    # degrees = torch.sparse.sum(sptensor, 0)
    degrees = torch.sum(sptensor, 0)
    D = degrees
    # D = D.to_dense()  # transfer D from sparse tensor to normal torch tensor
    D = torch.pow(D, -1)
    D = D.unsqueeze(dim=1)
    iteration = 2**order
    scale_list = list(2**i - 1 for i in range(order + 1))
    feature_p = feature
    sct_diffusion_list = []
    for i in range(iteration):
        D_inv_x = D * feature_p
        W_D_inv_x = torch.mm(sptensor, D_inv_x)
        # W_D_inv_x = torch.spmm(sptensor, D_inv_x)
        feature_p = 0.5 * feature_p + 0.5 * W_D_inv_x
        if i in scale_list:
            sct_diffusion_list += [
                feature_p,
            ]
    sct_feature1 = sct_diffusion_list[0] - sct_diffusion_list[1]
    sct_feature2 = sct_diffusion_list[1] - sct_diffusion_list[2]
    sct_feature3 = sct_diffusion_list[2] - sct_diffusion_list[3]
    return sct_feature1, sct_feature2, sct_feature3


###########

class GC_withres(Module):
    """
    res conv
    """

    def __init__(self, in_features, out_features, smooth,should_use_sparse=False):
        super(GC_withres, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smooth = smooth
        self.mlp = nn.Linear(in_features, out_features)
        self.should_use_sparse = should_use_sparse

    def set_should_use_sparse(self, should_use_sparse=None):
        if should_use_sparse is None:
            self.should_use_sparse = not self.should_use_sparse
        self.should_use_sparse = should_use_sparse
        return self.should_use_sparse


    def forward(self, input, adj, device="cuda"):
        if self.should_use_sparse:
            support = self.mlp(input)
            I_n = sp.eye(adj.size(0))
            I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
            A_gcn = adj + I_n
            degrees = torch.sparse.sum(A_gcn, 0)
            D = degrees
            D = D.to_dense()  # transfer D from sparse tensor to normal torch tensor
            D = torch.pow(D, -0.5)
            D = D.unsqueeze(dim=1)
            A_gcn_feature = support
            A_gcn_feature = torch.mul(A_gcn_feature, D)
            A_gcn_feature = torch.spmm(A_gcn, A_gcn_feature)
            A_gcn_feature = torch.mul(A_gcn_feature, D)
            output = A_gcn_feature * self.smooth + support
            output = output / (1 + self.smooth)
            return output
        support = self.mlp(input)
        # I_n = sp.eye(adj.size(0))
        I_n = torch.eye(adj.size(0)).to(device)
        # I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
        A_gcn = adj + I_n
        # degrees = torch.sparse.sum(A_gcn, 0)
        degrees = torch.sum(A_gcn, 0)
        D = degrees
        # D = D.to_dense()  # transfer D from sparse tensor to normal torch tensor
        D = torch.pow(D, -0.5)
        D = D.unsqueeze(dim=1)
        A_gcn_feature = support
        A_gcn_feature = torch.mul(A_gcn_feature, D)
        A_gcn_feature = torch.mm(A_gcn.to_dense(), A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature, D)
        output = A_gcn_feature * self.smooth + support
        output = output / (1 + self.smooth)
        return output

class SCTConv(torch.nn.Module):
    def __init__(self, hidden_dim, smooth, dropout, Withgres=False,shouls_use_sparse=False):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.a = Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        self.smoothlayer = Withgres  # turn on graph residual layer or not
        self.should_use_sparse = shouls_use_sparse
        self.gres = GC_withres(hidden_dim, hidden_dim, smooth=smooth,should_use_sparse=shouls_use_sparse)
        self.dropout = dropout

    def set_should_use_sparse(self, should_use_sparse=None):
        if should_use_sparse is None:
            self.should_use_sparse = not self.should_use_sparse
        self.should_use_sparse = should_use_sparse
        returned_should_use_sparse = self.gres.set_should_use_sparse(should_use_sparse)
        assert returned_should_use_sparse == self.should_use_sparse, "Mismatch in should_use_sparse settings!"
        return self.should_use_sparse

    def forward(self, X, adj, moment=1, device="cuda"):
        """
        Params
        ------
        adj [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        support0 = X
        N = support0.size()[0]
        h = support0
        gcn_diffusion_list = GCN_diffusion(adj, 3, support0, device=device,should_use_sparse=self.should_use_sparse)
        h_A = gcn_diffusion_list[0]
        h_A2 = gcn_diffusion_list[1]
        h_A3 = gcn_diffusion_list[2]

        h_A = nn.LeakyReLU()(h_A)
        h_A2 = nn.LeakyReLU()(h_A2)
        h_A3 = nn.LeakyReLU()(h_A3)

        h_sct1, h_sct2, h_sct3 = scattering_diffusion(adj, support0, should_use_sparse=self.should_use_sparse)
        h_sct1 = torch.abs(h_sct1) ** moment
        h_sct2 = torch.abs(h_sct2) ** moment
        h_sct3 = torch.abs(h_sct3) ** moment

        a_input_A = torch.hstack((h, h_A)).unsqueeze(1)
        a_input_A2 = torch.hstack((h, h_A2)).unsqueeze(1)
        a_input_A3 = torch.hstack((h, h_A3)).unsqueeze(1)
        a_input_sct1 = torch.hstack((h, h_sct1)).unsqueeze(1)
        a_input_sct2 = torch.hstack((h, h_sct2)).unsqueeze(1)
        a_input_sct3 = torch.hstack((h, h_sct3)).unsqueeze(1)

        a_input = torch.cat(
            (
                a_input_A,
                a_input_A2,
                a_input_A3,
                a_input_sct1,
                a_input_sct2,
                a_input_sct3,
            ),
            1,
        ).view(N, 6, -1)
        # GATV2
        e = torch.matmul(torch.nn.functional.relu(a_input), self.a).squeeze(2)
        attention = F.softmax(e, dim=1).view(N, 6, -1)
        #        h_all = torch.cat((h_A.unsqueeze(dim=2),h_A2.unsqueeze(dim=2),h_A3.unsqueeze(dim=2),h_sct1.unsqueeze(dim=2),h_sct2.unsqueeze(dim=2),h_sct3.unsqueeze(dim=2)),dim=2).view(N,6, -1)
        h_all = torch.cat(
            (
                h_A.unsqueeze(dim=1),
                h_A2.unsqueeze(dim=1),
                h_A3.unsqueeze(dim=1),
                h_sct1.unsqueeze(dim=1),
                h_sct2.unsqueeze(dim=1),
                h_sct3.unsqueeze(dim=1),
            ),
            dim=1,
        )
        h_prime = torch.mul(attention, h_all)  # element eise product
        h_prime = torch.mean(h_prime, 1)
        if self.smoothlayer:
            h_prime = self.gres(h_prime, adj, device)
        else:
            pass
        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X

class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        Withgres=False,
        smooth=0.0,
        repeated=1,
        should_use_sparse=False
    ):
        super().__init__()
        self.dropout = dropout
        self.smooth = smooth
        self.repeated = repeated
        self.should_use_sparse = should_use_sparse
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SCTConv(hidden_dim, self.smooth, self.dropout, Withgres,shouls_use_sparse=should_use_sparse))
        self.mlp1 = Linear(hidden_dim * (1 + n_layers*repeated), hidden_dim)
        self.mlp2 = Linear(hidden_dim, output_dim)

    def set_should_use_sparse(self, should_use_sparse=None):
        if should_use_sparse is None:
            self.should_use_sparse = not self.should_use_sparse
        self.should_use_sparse = should_use_sparse
        for layer in self.convs:
            returned_should_use_sparse = layer.set_should_use_sparse(should_use_sparse)
            assert returned_should_use_sparse == self.should_use_sparse, "Mismatch in should_use_sparse settings!"
        return self.should_use_sparse

    def forward(self, X, adj, moment=1, device="cuda", return_hidden=False):
        numnodes = X.size(0)
        scale = np.sqrt(numnodes)  # for graph norm
        X = X / scale
        X = self.in_proj(X)
        hidden_states = X
        hidden = []
        # hidden.append(X)
        for _ in range(self.repeated):
            for layer in self.convs:
                X = layer(X, adj, moment=moment, device=device)
                # normalize
                X = X / scale
                hidden_states = torch.cat([hidden_states, X], dim=1)
                hidden.append(X)
        X = hidden_states
        # hidden.append(X)  # -4
        X = self.mlp1(X)
        # hidden.append(X)  # -3
        X = F.leaky_relu(X)
        # hidden.append(X)  # -2
        X = self.mlp2(X)
        # hidden.append(X)  # -1
        maxval = torch.max(X)
        minval = torch.min(X)
        X = (X - minval) / (maxval + 1e-6 - minval)
        if return_hidden:
            return X, hidden
        return X


# scattering model
class ScatteringNoFeaturesModel:
    def __init__(
        self,
        pth_file: str = None,
        device: str = "cuda",
        model_state_dict_key="model_state_dict",
        use_networkx=False,
        should_use_sparse=False,
        repeated=1,
        n_layers=4,
        GNN_kwargs={},
    ):
        self.device = device
        self.checkpoint = None

        if pth_file is not None:
            checkpoint = torch.load(pth_file, map_location=torch.device(self.device))
            self.checkpoint = checkpoint
            repeated = checkpoint.get("repeated",repeated)
            n_layers = checkpoint.get("n_layers",n_layers)

        if len(GNN_kwargs) > 0:
            self.model = GNN(**GNN_kwargs)
        else:
            self.model = GNN(
                input_dim=3,
                hidden_dim=8,
                output_dim=1,
                n_layers=n_layers,
                dropout=0,
                Withgres=False,
                smooth=0.1,
                should_use_sparse=should_use_sparse,
                repeated=repeated,
            )
        self.model.train()
        if self.checkpoint is not None:
            if model_state_dict_key:
                checkpoint = checkpoint[model_state_dict_key]
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        self.model.to(self.device)
        self.use_networkx = use_networkx

        if should_use_sparse is not None:
            self.set_should_use_sparse(should_use_sparse)

    def set_should_use_sparse(self, should_use_sparse=None):
        if should_use_sparse is None:
            self.should_use_sparse = not self.should_use_sparse
        self.should_use_sparse = should_use_sparse
        returned_sparse = self.model.set_should_use_sparse(should_use_sparse)
        assert self.should_use_sparse == returned_sparse, "Mismatch in should_use_sparse settings!"
        return self.should_use_sparse

    def get_gnn(self):
        return self.model

    def get_optimizer(self, lr=4e-4, weight_decay=0.0):
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=0.0)
        epoch = 0
        batch = 0
        if self.checkpoint is not None:
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.model.cuda()
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cuda")
            epoch = self.checkpoint["epoch"] + 1
            batch = self.checkpoint["batch"] + 1
            batch = 0
            print(f"Resuming training from epoch {epoch}, batch {batch}")
        return self.model, optimizer, epoch, batch

    def compute_features(self, G, device=None):
        device = device if device is not None else self.device

        n_nodes = len(G)
        feature_vector = [[1, 1, 1] for _ in range(n_nodes)]
        feature_vector = np.array(feature_vector)
        return torch.FloatTensor(feature_vector).to(device)

    def compute_prob(self, G, to_numpy=True, return_hidden=False):
        features = self.compute_features(G)
        if self.use_networkx:
            data = from_networkx(G)
            adj = to_scipy_sparse_matrix(
                edge_index=data.edge_index, num_nodes=data.num_nodes
            )
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            adj = adj.to_dense()
        else:
            adj = torch.from_numpy(G).float()

        adj = adj.to(self.device)
        if features.shape[0] != adj.shape[0]:
            print("Error: features and adj do not match")
        if return_hidden:
            prob, hidden = self.model(
                features, adj, moment=1, device=self.device, return_hidden=True
            )
        else:
            prob = self.model(features, adj, moment=1, device=self.device)
            hidden = None
        if to_numpy:
            prob = prob.detach().cpu().numpy().flatten()
            if hidden:
                hidden = [h.detach().cpu().numpy() for h in hidden]
        if return_hidden:
            return prob, hidden
        return prob

    def least_probable_removal(
        self,
        G,
        compute_once=False,
        reversed_order=False,
        use_1st_pc=False,
        return_history=False,
        fix_by_spearman=False,
    ):
        def compute_prob(H):
            if use_1st_pc:
                _, hidden = self.compute_prob(H, return_hidden=True)
                embedding = hidden[-1]
                if np.isnan(embedding).any():
                    print("nan embedding")
                pca = PCA(n_components=1)
                try:
                    pca.fit(embedding)
                except Exception as e:
                    print("PCA fit error:", e)
                pc1 = (embedding - pca.mean_) @ (pca.components_.T)
                probs = pc1
            else:
                probs = self.compute_prob(H)
            if reversed_order:
                probs = 1 - probs
            return probs

        return least_probable_removal_numpy(
            G,
            compute_prob,
            compute_once=compute_once,
            return_history=return_history,
            fix_by_spearman=fix_by_spearman,
        )
