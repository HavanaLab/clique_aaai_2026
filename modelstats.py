from model import GNN
from erdos.erdos_model import clique_MPNN


def param_stats(model, modules):
    total, trainable, mem = 0, 0, 0
    for name, mod in model.named_modules():
        if name in modules:
            for p in mod.parameters():
                n = p.numel()
                total += n
                mem += n * p.element_size()
                if p.requires_grad:
                    trainable += n
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Memory:           {mem / 1024**2:.2f} MiB")

repeated = 1
n_layers = 4
models_names_pair = [
    ("NeuCLiqueMPNN", GNN(input_dim=3, hidden_dim=8, output_dim=1, n_layers=n_layers, dropout=0, Withgres=False,
                          smooth=0.1, should_use_sparse=False, repeated=repeated
                          )),
    ("Erdos", clique_MPNN(4, 64, 1, 1)),
]

for name, model in models_names_pair:
    print(f"\n --------------Model {name} Start-----------\n")
    neurosat_modules_to_check = {name for name, mod in model.named_modules()}
    print(neurosat_modules_to_check)
    param_stats(model, neurosat_modules_to_check)
    print(f"\n --------------Model {name} Finish-----------\n")

