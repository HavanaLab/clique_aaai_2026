import os.path

from sklearn.decomposition import PCA
from erdos.erdos_model import ErdosModel
from evaluate.evaluate import get_calculated_rows
from generate_difficult_instances import load_and_parse_from_jsonl
from graphs.graph_utils_numpy import get_degrees, is_clique

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def least_probable_removal_numpy(
    adj,
    compute_prob,
    compute_once=False,
    return_history=False,
    fix_by_spearman=False,
    should_remove_isolated_nodes=False,
):
    history = {
        "removed_nodes": [],
        # "iterations": [],
        "graph_sizes": [],
        # "spearman": [],
        "masks": [],
        "degrees": [],
        "degrees_quantiles": [],
        "pc1": [],
        "probs": [],
        "probs_quantiles": [],
        "pca_layer_0": [],
        "pca_layer_1": [],
        "pca_layer_2": [],
        "pca_layer_3": [],
        "pca_quantiles_layer_0": [],
        "pca_quantiles_layer_1": [],
        "pca_quantiles_layer_2": [],
        "pca_quantiles_layer_3": [],
    }

    R = []
    if adj.sum() == 0:
        return np.arange([])

    removal_phase_iteration = 0

    n = len(adj)
    n_range = np.arange(n)
    mask = np.ones(n).astype(np.True_)
    subgraph = adj
    subgraph_probs_first, hidden = compute_prob(subgraph)
    for layer, h in enumerate(hidden):
        embedding = h.cpu().detach().numpy()
        pca = PCA(n_components=1)
        pca.fit(embedding)
        pc1 = (embedding - pca.mean_) @ (pca.components_.T)
        pc1 = pc1.flatten()
        pc1_quantiles = np.digitize(
            pc1, bins=np.quantile(pc1, [0.25, 0.5, 0.75]), right=False
        )
        history[f"pca_quantiles_layer_{layer}"].append(pc1_quantiles)
        history[f"pca_layer_{layer}"].append(pc1)
    history["masks"].append(mask.tolist())
    history["pc1"].append(pc1)

    probs_quantiles_first = np.digitize(
        subgraph_probs_first,
        bins=np.quantile(subgraph_probs_first, [0.25, 0.5, 0.75]),
        right=False,
    )
    history["probs_quantiles"].append(probs_quantiles_first)

    history["probs"].append(subgraph_probs_first)
    degrees = get_degrees(adj)
    history["degrees"].append(degrees)
    degrees_quantiles = np.digitize(
        degrees, bins=np.quantile(degrees, [0.25, 0.5, 0.75]), right=False
    )
    history["degrees_quantiles"].append(degrees_quantiles)
    history["graph_sizes"].append(len(subgraph))
    while not is_clique(subgraph):
        if removal_phase_iteration > 9:
            break
        subgraph = adj[mask, :][:, mask]
        first_iteration = len(R) == 0

        if any(subgraph.sum(axis=0) == 0) and should_remove_isolated_nodes:
            # remove isolated nodes from subgraph and mask
            nodes_to_remove = np.where(subgraph.sum(axis=0) == 0)[0]
            nodes_to_remove_in_original = n_range[mask][nodes_to_remove]
            mask[nodes_to_remove_in_original] = False
            subgraph = adj[mask, :][:, mask]
            R.extend(nodes_to_remove_in_original)
            continue
            # add isolated nodes to R
        if compute_once or first_iteration:
            # remove v from subgraph_probs
            subgraph_probs = subgraph_probs_first[mask]
        else:
            subgraph_probs, hidden = compute_prob(subgraph)
        # embedding = hidden[-1].detach().numpy()
        removal_phase_iteration += 1
        for layer, h in enumerate(hidden):
            embedding = h.cpu().detach().numpy()
            pca = PCA(n_components=1)
            pca.fit(embedding)
            pc1 = (embedding - pca.mean_) @ (pca.components_.T)
            pc1 = pc1.flatten()
            pc1_quantiles = np.digitize(
                pc1, bins=np.quantile(pc1, [0.25, 0.5, 0.75]), right=False
            )
            history[f"pca_quantiles_layer_{layer}"].append(pc1_quantiles)
            history[f"pca_layer_{layer}"].append(pc1)
        degrees = get_degrees(subgraph)


        history["probs"].append(subgraph_probs)
        history["degrees"].append(degrees)
        history["graph_sizes"].append(len(subgraph))

        v = n_range[mask][subgraph_probs.argmin()]
        R.append(v)
        mask[v] = False
        subgraph = adj[mask, :][:, mask]
        history["masks"].append(mask.tolist())

    # reverse R
    R.reverse()

    if return_history:
        R.reverse()
        history["removed_nodes"] = R
        return n_range[mask], history
    return n_range[mask]


def least_probable_removal(
    model,
    adj_matrix,
    compute_once=False,
    should_remove_isolated_nodes=False,
    fix_by_spearman=False,
):
    probs = model.compute_prob(adj_matrix)
    probs = probs.squeeze()

    def compute_prob(H):
        probs, hidden = model.compute_prob(H, return_hidden=True)
        probs = probs.detach().numpy().flatten()
        return probs, hidden

    return least_probable_removal_numpy(
        adj_matrix,
        compute_prob,
        compute_once=compute_once,
        should_remove_isolated_nodes=should_remove_isolated_nodes,
        fix_by_spearman=fix_by_spearman,
        return_history=True,
    )


def unmask(mask, smaller_array):
    """
    smaller_array might not be smaller :p
    """
    ret = [_ for _ in mask]
    smaller_index = 0
    for i, item in enumerate(mask):
        if i >= len(smaller_array):
            ret[i] = None
        elif item:
            ret[i] = smaller_array[smaller_index]
            smaller_index += 1
        else:
            ret[i] = None
    return ret


import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


model_names = [
    "com-youtube_original",
    "com-orkut_original",
    "imdb_binary_original",
    "collab_original",
    "n500_easy_original",
    "n500_medium_original",
    "n500_hard_original",
    "tiwtter_original",
    "n1000_easy_original",
    "n1000_medium_original",
    "n1000_hard_original",
]

datasets = [
    "collab",
    "imdb_binary",
    "twitter",
    "com-youtube",
    "com-orkut",
    "facebook",
    "n1000_easy_instances",
    "n1000_medium_instances",
    "n1000_hard_instances",
    "n500_easy_instances",
    "n500_medium_instances",
    "n500_hard_instances",
]

for dataset_name in datasets:
    for model_name in model_names:
        model_path = f"../models/erdos/{model_name}.pth"
        model = ErdosModel(model_path, device="cpu")
        model.model.eval()
        dataset_path = f"../datasets/{dataset_name}.jsonl"
        dataset = load_and_parse_from_jsonl(dataset_path, to_networkx=False)

        model_name = model_name.replace("_instances", "")
        model_name = model_name.replace("_original", "")
        print(f"Running model:{model_name} on {dataset_name}")

        output_base = "../results/erdos ldr first 10 iteration quantiles clique"
        output_path = os.path.join(output_base,f"{model_name}_on_{dataset_name}.jsonl")
        os.makedirs(output_base, exist_ok=True)

        calculated_rows = get_calculated_rows(output_path, unique=False, cached=False)
        with open(output_path, "a", 1) as f:
            for i, instance in enumerate(dataset):
                if i in calculated_rows:
                    print("skippintg", i)
                    continue
                print("calc", i)
                G = instance["G"]
                sparse_similarity_matrix = csr_matrix(G)
                # clique = get_max_clique(G_nx)
                clique = instance.get("clique", None) or instance.get("clique_", None)
                clique_size = len(clique) if clique is not None else instance.get("max_clique_size", -1)
                ldr_answer, history = least_probable_removal(
                    model, G, should_remove_isolated_nodes=True
                )
                history["clique"] = clique
                history["clique_size"] = clique_size

                clique_members_mask = np.zeros(len(G), dtype=bool)
                clique_members_mask[clique] = True
                for layer in range(4):
                    for iteration, pc_quantiles in enumerate(
                        history[f"pca_quantiles_layer_{layer}"]
                    ):
                        unmasked_pc_quantiles = unmask(
                            clique_members_mask, pc_quantiles
                        )
                        clique_size = clique_size
                        non_clique_size = len(G) - clique_size
                        number_of_clique_members_in_q0 = sum(
                            np.array(unmasked_pc_quantiles)[clique_members_mask] == 0
                        )
                        number_of_clique_members_in_q1 = sum(
                            np.array(unmasked_pc_quantiles)[clique_members_mask] == 1
                        )
                        number_of_clique_members_in_q2 = sum(
                            np.array(unmasked_pc_quantiles)[clique_members_mask] == 2
                        )
                        number_of_clique_members_in_q3 = sum(
                            np.array(unmasked_pc_quantiles)[clique_members_mask] == 3
                        )

                        q0_size = sum(pc_quantiles == 0)
                        q1_size = sum(pc_quantiles == 1)
                        q2_size = sum(pc_quantiles == 2)
                        q3_size = sum(pc_quantiles == 3)

                        number_of_non_clique_members_in_q0 = (
                            q0_size - number_of_clique_members_in_q0
                        )
                        number_of_non_clique_members_in_q1 = (
                            q1_size - number_of_clique_members_in_q1
                        )
                        number_of_non_clique_members_in_q2 = (
                            q2_size - number_of_clique_members_in_q2
                        )
                        number_of_non_clique_members_in_q3 = (
                            q3_size - number_of_clique_members_in_q3
                        )

                        history[f"iter_{iteration}_layer_{layer}_q0_pca_clique"] = (
                            number_of_clique_members_in_q0 / clique_size
                        )
                        history[f"iter_{iteration}_layer_{layer}_q0_pca_non_clique"] = (
                            (number_of_non_clique_members_in_q0 / non_clique_size)
                            if non_clique_size > 0
                            else 0
                        )

                for iteration, degree_quantile in enumerate(
                    history["degrees_quantiles"]
                ):
                    unmasked_degree_quantile = unmask(
                        clique_members_mask, degree_quantile
                    )
                    clique_size = clique_size
                    non_clique_size = len(G) - clique_size
                    number_of_clique_members_in_q0 = sum(
                        np.array(unmasked_degree_quantile)[clique_members_mask] == 0
                    )
                    number_of_clique_members_in_q1 = sum(
                        np.array(unmasked_degree_quantile)[clique_members_mask] == 1
                    )
                    number_of_clique_members_in_q2 = sum(
                        np.array(unmasked_degree_quantile)[clique_members_mask] == 2
                    )
                    number_of_clique_members_in_q3 = sum(
                        np.array(unmasked_degree_quantile)[clique_members_mask] == 3
                    )

                    q0_size = sum(degree_quantile == 0)
                    q1_size = sum(degree_quantile == 1)
                    q2_size = sum(degree_quantile == 2)
                    q3_size = sum(degree_quantile == 3)

                    number_of_non_clique_members_in_q0 = (
                        q0_size - number_of_clique_members_in_q0
                    )
                    number_of_non_clique_members_in_q1 = (
                        q1_size - number_of_clique_members_in_q1
                    )
                    number_of_non_clique_members_in_q2 = (
                        q2_size - number_of_clique_members_in_q2
                    )
                    number_of_non_clique_members_in_q3 = (
                        q3_size - number_of_clique_members_in_q3
                    )

                    history[f"iter_{iteration}_q0_clique"] = (
                        number_of_clique_members_in_q0 / clique_size
                    )
                    history[f"iter_{iteration}_q0__non_clique"] = (
                        (number_of_non_clique_members_in_q0 / non_clique_size)
                        if non_clique_size > 0
                        else 0
                    )

                for iteration, probs_quantile in enumerate(history["probs_quantiles"]):
                    unmasked_probs_quantile = unmask(
                        clique_members_mask, probs_quantile
                    )
                    clique_size = clique_size
                    non_clique_size = len(G) - clique_size
                    number_of_clique_members_in_q0 = sum(
                        np.array(unmasked_probs_quantile)[clique_members_mask] == 0
                    )
                    number_of_clique_members_in_q1 = sum(
                        np.array(unmasked_probs_quantile)[clique_members_mask] == 1
                    )
                    number_of_clique_members_in_q2 = sum(
                        np.array(unmasked_probs_quantile)[clique_members_mask] == 2
                    )
                    number_of_clique_members_in_q3 = sum(
                        np.array(unmasked_probs_quantile)[clique_members_mask] == 3
                    )

                    q0_size = sum(probs_quantile == 0)
                    q1_size = sum(probs_quantile == 1)
                    q2_size = sum(probs_quantile == 2)
                    q3_size = sum(probs_quantile == 3)

                    number_of_non_clique_members_in_q0 = (
                        q0_size - number_of_clique_members_in_q0
                    )
                    number_of_non_clique_members_in_q1 = (
                        q1_size - number_of_clique_members_in_q1
                    )
                    number_of_non_clique_members_in_q2 = (
                        q2_size - number_of_clique_members_in_q2
                    )
                    number_of_non_clique_members_in_q3 = (
                        q3_size - number_of_clique_members_in_q3
                    )

                    history[f"iter_probs_{iteration}_q0_clique"] = (
                        number_of_clique_members_in_q0 / clique_size
                    )
                    history[f"iter_probs_{iteration}_q0__non_clique"] = (
                        (number_of_non_clique_members_in_q0 / non_clique_size)
                        if non_clique_size > 0
                        else 0
                    )
                json.dump(history, f, cls=NpEncoder)
                f.write("\n")
