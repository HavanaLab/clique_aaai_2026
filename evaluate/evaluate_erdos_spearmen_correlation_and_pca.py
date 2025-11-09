import os.path
from itertools import product
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats

import json
from scipy.sparse import csr_matrix
import networkx as nx

import pandas as pd

from evaluate import EvaluateDataset, get_calculated_rows, handle_row

from graphs.graph_utils_numpy import get_degrees

from erdos.erdos_model import ErdosModel


def compute_spearman(x, y):
    return stats.spearmanr(x, y)



def compute_spearman_for_G_degree_quantiles(G, model, layers=[-5]):
    prob, hidden = model.compute_prob(G, return_hidden=True)
    prob = prob.flatten().detach().numpy()
    sparse_similarity_matrix = csr_matrix(G)
    G = nx.from_scipy_sparse_array(sparse_similarity_matrix)
    G = G.to_undirected()
    degrees = np.array([d[1] for d in nx.degree(G)])

    nodes_quantiles = generate_node_degree_quantiles(degrees)
    nodes_quantiles_masks = [generate_mask_array(G, nodes) for nodes in nodes_quantiles]

    res = {}
    layers = range(len(hidden))
    for layer, (mask_index, node_mask) in product(
        layers, enumerate(nodes_quantiles_masks)
    ):
        if sum(node_mask) == 0:
            continue
        embedding = hidden[layer].detach().numpy()
        pca = PCA(n_components=3)
        dim_reduced_embedding = pca.fit_transform(embedding)
        pc1_explained = float(pca.explained_variance_ratio_[0])
        pc2_explained = float(pca.explained_variance_ratio_[1])
        pc3_explained = float(pca.explained_variance_ratio_[2])
        pc_explained = [pc1_explained, pc2_explained, pc3_explained]
        spearman_prop_degree = compute_spearman_between_property_and_pc(
            dim_reduced_embedding[node_mask], degrees[node_mask]
        )

        res.update(
            {
                f"pc1_explained_layer_{layer}": pc_explained[0],
                f"pc2_explained_layer_{layer}": pc_explained[1],
                f"pc3_explained_layer_{layer}": pc_explained[2],
                f"spearman_q{mask_index}_layer_{layer}_between_degree_and_pc1": spearman_prop_degree[
                    0
                ][
                    0
                ],
                f"spearman_q{mask_index}_layer_{layer}_between_degree_and_pc1_p": spearman_prop_degree[
                    0
                ][
                    1
                ],
                f"spearman_q{mask_index}_layer_{layer}_between_degree_and_pc2": spearman_prop_degree[
                    1
                ][
                    0
                ],
                f"spearman_q{mask_index}_layer_{layer}_between_degree_and_pc2_p": spearman_prop_degree[
                    1
                ][
                    1
                ],
                f"spearman_q{mask_index}_layer_{layer}_between_degree_and_pc3": spearman_prop_degree[
                    2
                ][
                    0
                ],
                f"spearman_q{mask_index}_layer_{layer}_between_degree_and_pc3_p": spearman_prop_degree[
                    2
                ][
                    1
                ],
            }
        )
    return res


###
def generate_node_degree_quantiles(degrees, quantiles=[0.25, 0.5, 0.75]):
    nodes = np.arange(len(degrees))
    degree_quantiles = np.quantile(degrees, quantiles)
    nodes_quantiles = []
    for i in range(len(quantiles)):
        if i == 0:
            nodes_quantiles.append(
                [node for node in nodes if degrees[node] <= degree_quantiles[i]]
            )
        else:
            nodes_quantiles.append(
                [
                    node
                    for node in nodes
                    if degree_quantiles[i - 1] < degrees[node] <= degree_quantiles[i]
                ]
            )
    nodes_quantiles.append(
        [node for node in nodes if degree_quantiles[-1] < degrees[node]]
    )
    return nodes_quantiles


def generate_mask_array(G, nodes):
    mask_array = np.zeros((len(G.nodes())), dtype=bool)
    mask_array[nodes] = True
    return mask_array


def compute_spearman_between_property_and_pc(embedding, prop):

    spearman_between_prop_and_pc2 = compute_spearman(prop, embedding[:, 1])

    spearman_between_prop_and_pc1 = compute_spearman(prop, embedding[:, 0])

    spearman_between_prop_and_pc3 = compute_spearman(prop, embedding[:, 2])
    spearman_prop = [
        spearman_between_prop_and_pc1,
        spearman_between_prop_and_pc2,
        spearman_between_prop_and_pc3,
    ]
    return spearman_prop


###


def compute_spearman_for_G(G, model, layers=[-5]):
    prob, hidden = model.compute_prob(G, return_hidden=True)
    prob = prob.flatten().detach().numpy()
    degrees = get_degrees(G)
    res = {}
    layers = range(len(hidden))
    spearman_between_prob_and_degree = compute_spearman(prob, degrees)

    for layer in layers:
        embedding = hidden[layer].detach().numpy()
        pca = PCA(n_components=3)
        dim_reduced_embedding = pca.fit_transform(embedding)
        pc1_explained = float(pca.explained_variance_ratio_[0])
        pc2_explained = float(pca.explained_variance_ratio_[1])
        pc3_explained = float(pca.explained_variance_ratio_[2])

        spearman_between_degree_and_pc2 = compute_spearman(
            degrees, dim_reduced_embedding[:, 1]
        )

        spearman_between_degree_and_pc1 = compute_spearman(
            degrees, dim_reduced_embedding[:, 0]
        )

        spearman_between_degree_and_pc3 = compute_spearman(
            degrees, dim_reduced_embedding[:, 2]
        )

        spearman_between_prob_and_pc1 = compute_spearman(dim_reduced_embedding[:, 0], prob)


        res.update(
            {
                f"pc1_explained_layer_{layer}": pc1_explained,
                f"pc2_explained_layer_{layer}": pc2_explained,
                f"pc3_explained_layer_{layer}": pc3_explained,
                f"spearman_layer_{layer}_between_prob_and_pc1": spearman_between_prob_and_pc1[
                    0
                ],
                f"spearman_layer_{layer}_between_prob_and_pc1_p": spearman_between_prob_and_pc1[
                    1
                ],
                f"spearman_layer_{layer}_between_degree_and_pc3": spearman_between_degree_and_pc3[
                    0
                ],
                f"spearman_layer_{layer}_between_degree_and_pc3_p": spearman_between_degree_and_pc3[
                    1
                ],
                f"spearman_layer_{layer}_between_degree_and_pc2": spearman_between_degree_and_pc2[
                    0
                ],
                f"spearman_layer_{layer}_between_degree_and_pc2_p": spearman_between_degree_and_pc2[
                    1
                ],
                f"spearman_layer_{layer}_between_degree_and_pc1": spearman_between_degree_and_pc1[
                    0
                ],
                f"spearman_layer_{layer}_between_degree_and_pc1_p": spearman_between_degree_and_pc1[
                    1
                ],
            }
        )
    res.update({ f"spearman_between_prob_and_degree": spearman_between_prob_and_degree[
                    0
                ],
                f"spearman_between_prob_and_degree_p": spearman_between_prob_and_degree[
                    1
                ]
                 }
               )
    return res


def generate_model(model_path, model_name, device="cpu"):
    model = ErdosModel(model_path, device=device)
    model.model.eval()
    return model


datasets_paths = [
    "n500_easy_instances",
    "n500_medium_instances",
    "n500_hard_instances",
    "n1000_easy_instances",
    "n1000_medium_instances",
    "n1000_hard_instances",
    "twitter",
    "collab",
    "imdb_binary",
    "com-youtube",
    "facebook",
    "com-orkut",
]

model_paths = [
    "twitter_original"
]


def evaluate_graph_func(instance, model):
    G = instance["G"]
    result_row1 = compute_spearman_for_G(G, model)
    result_row2 = compute_spearman_for_G_degree_quantiles(G, model)
    result_row = {**result_row1, **result_row2}
    return result_row



def main():
    for di, dataset_name in enumerate(datasets_paths):
        for mi, model_name in enumerate(model_paths):
            dataset_path = f"../datasets/{dataset_name}.jsonl"
            model_prefix = f"../models/erdos/{model_name}.pth"
            output_base = "../results/erdos correlation and pca"
            output_path = os.path.join(output_base, f"ranges_{model_name}_on_{dataset_name}_eval_results.jsonl")
            os.makedirs(output_base, exist_ok=True)

            model = generate_model(model_prefix, model_name, device="cpu")
            eval_dataset = EvaluateDataset([], input_path=dataset_path, output_path=output_path)

            calculated_rows = get_calculated_rows(output_path, unique=True, cached=False)
            print(f"calculated_rows: {len(calculated_rows)}")
            for i, row in enumerate(eval_dataset.dataset):
                if i in calculated_rows:
                    print(f"Skipping row {i} already calculated")
                    continue
                handle_row(output_path, row, i, model, evaluate_graph_func_=evaluate_graph_func)


if __name__ == "__main__":
    main()
