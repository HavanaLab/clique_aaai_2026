from erdos.jsonl_dataset import JSONLDataset
from erdos.erdos_model import clique_MPNN, train
from torch.optim import Adam
import torch_geometric.transforms as T

# dataset_name = "n1000_hard_instances"
# dataset_name = "twitter"
dataset_name = "n1000_medium_instances"
# dataset_name = "n1000_easy_instances"

if __name__ == "__main__":
    print(f"Training on {dataset_name}")
    dataset = JSONLDataset(
        file_path=f"datasets/{dataset_name}.jsonl", transform=T.ToSparseTensor()
    )
    checkpoint_path = f"models/erdos_1_layers/new_erdos_neu_clique_MPNN_{dataset_name}.pth"
    model = clique_MPNN(4, 64, 1, 1)
    model.to("cuda").reset_parameters()

    learning_rate = 0.001
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train(
        dataset, model, optimizer, epochs=100, batch_size=32, checkpoint=checkpoint_path
    )
