# Consolidated most of the code from https://github.com/yimengmin/geometricscatteringmaximalclique
# onto this file.

import torch
import torch.nn

from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import from_networkx, dense_to_sparse
from torch_geometric.data import Data

from generate_difficult_instances import load_and_parse_from_json
from model import sparse_mx_to_torch_sparse_tensor, ScatteringNoFeaturesModel, compute_loss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
###########

def numpy_to_torch_sparse_tensor(G):
    G = torch.from_numpy(G)
    edge_index, edge_attr = dense_to_sparse(G)
    N = G.size(0)
    x = torch.ones((N, 1))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def compute_prob(G, model, device="cpu"):
    gnn = model.get_gnn()
    features = model.compute_features(G, device=device)
    # data = from_networkx(G)
    data = numpy_to_torch_sparse_tensor(G)
    adj = to_scipy_sparse_matrix(edge_index=data.edge_index, num_nodes=data.num_nodes)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to(device)
    previous_mode = gnn.training
    gnn.eval()
    gnn.to(device)
    if features.shape[0] != adj.shape[0]:
        print("Error: features and adj do not match")
    prob = gnn(features, adj, moment=1, device=device)
    gnn.train(previous_mode)
    return prob, data



def train_model(
    dataset_path,
    checkpoint_path,
    start_from_checkpoint=None,
    epochs=15,
    training_size=900,
    batch_size=100,
    lr=4e-4,
    penalty_coefficient=0.005,
    n_layers=4,
    repeated=1
):
    model = ScatteringNoFeaturesModel(pth_file=start_from_checkpoint, use_networkx=False, should_use_sparse=True)
    gnn, optimizer, epoch, batch = model.get_optimizer(lr=lr)    # train for 15 epochs on 900 first graphs
    for i in range(epoch, epochs):
        print(f"Epoch {i} started")
        batch_counter = 0
        batch_loss = 0.0
        dataset = load_and_parse_from_json(dataset_path, lines=True, to_networkx=False) #, skip = 0 if epoch != i else batch*batch_size)
        for j, training_instance in enumerate(dataset):
            if batch > j // batch_size and i == epoch:
                print(f"skipping {j} already trained")
                continue
            if j >= training_size:
                break
            print(f"Epoch {i}, Graph {j} started")
            G = training_instance["G"]

            prob, data = compute_prob(G, model, device="cuda")
            loss = compute_loss(
                data.edge_index,
                prob,
                penalty_coefficient=penalty_coefficient,
            ).cuda()
            batch_loss += loss
            batch_counter += 1
            if batch_counter == batch_size:
                batch_loss /= batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1)
                optimizer.step()
                torch.save(
                    {
                        "epoch": i,
                        "batch": j // batch_size,
                        "model_state_dict": gnn.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "repeated": repeated,
                        "n_layers": n_layers,
                    },
                    checkpoint_path,
                )
                torch.save(
                    {
                        "epoch": i,
                        "batch": j // batch_size,
                        "model_state_dict": gnn.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "repeated": repeated,
                        "n_layers": n_layers,
                    },
                    f"{checkpoint_path}{i}",
                )
                print(f"Epoch {i}, Batch {j//batch_size} completed")
                print("Batch Loss:", batch_loss.item())
                batch_loss = 0.0
                batch_counter = 0
        print(f"Epoch {i} completed")


if __name__ == "__main__":
    train_model(
        "datasets/n1000_medium_instances.jsonl",
        "models/scattering2/n1000_medium_instances_15_epochs_wc.pth",
        epochs=15,
        training_size=900,
        batch_size=300,
        lr=4e-4,
    )
    # train_model(
    #     "n1000_medium_instances.jsonl",
    #     "models/no_features/n1000_medium_instances_15_epochs_wc.pth",
    #     epochs=15,
    #     training_size=900,
    #     batch_size=300,
    #     lr=4e-4,
    # )
    # train_model(
    #     "datasets/n1000_hard_instances.jsonl",
    #     "models/no_features/n1000_hard_instances_15_epochs_wc.pth",
    #     epochs=15,
    #     training_size=900,
    #     batch_size=300,
    #     lr=4e-4,
    # )
    # train_model(
    #     "datasets/twitter.jsonl",
    #     "models/no_features/twitter_15_epochs_wc.pth",
    #     epochs=15,
    #     training_size=900,
    #     batch_size=300,
    #     lr=4e-4,
    # )
    # train_model(
    #     "datasets/n1000_gnp_instances.jsonl",
    #     "models/no_features/n1000_gnp_instances_15_epochs_wc.pth",
    #     epochs=15,
    #     training_size=900,
    #     batch_size=300,
    #     lr=4e-4,
    # )
