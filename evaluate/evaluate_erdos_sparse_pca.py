import json
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from evaluate import Evaluate, EvaluateDataset, handle_row, get_calculated_rows
from evaluate_erdos_spearmen_correlation_and_pca import compute_spearman
from graphs.graph_utils_numpy import get_degrees
from model import ScatteringNoFeaturesModel
from erdos.erdos_model import ErdosModel

from utils.pca import generate_sample_covariance, \
    run_sklearn_sparse_pca
from sklearn.decomposition import SparsePCA





def evaluate_graph_func(instance, evals):
    G = instance["G"]
    max_clique_size = instance["max_clique_size"]
    result_row = {}
    for eval in evals:
        print(f"evaluating {eval.name}")
        ev_result = eval(G)
        ratios, ratios_classic, ratio_by_adj = ev_result
        for i, (r, rc, rba) in enumerate(zip(ratios, ratios_classic, ratio_by_adj)):
            result_row[f"{eval.name}_model_{i}"] = r
            result_row[f"{eval.name}_subgraph_{i}"] = rc
            result_row[f"{eval.name}_adj_{i}"] = rba
    return result_row

def generate_models_evaluates(
    model_path, model_name, device="cuda", thresholdloopnodes=300
):
    model = ErdosModel(model_path, device=device)
    return [
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G,
        #         use_1st_pc=True,
        #         fix_by_spearman=True,
        #     ),
        #     f"ldr_pca_spearman",
        # ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G,
        #         use_1st_pc=True,
        #         fix_by_spearman=True,
        #         compute_once=True,
        #     ),
        #     f"ldr_pca_spearman_once",
        # ),
        # Evaluate(
        #     lambda G: model.predict_and_decode(
        #         G, thresholdloopnodes=thresholdloopnodes
        #     ),
        #     f"{model_name}_decode_k_10",
        # ),
        # Evaluate(
        #     lambda G: model.predict_and_decode(
        #         G, num_walkers=20, thresholdloopnodes=thresholdloopnodes
        #     ),
        #     f"decode_k_20",
        # ),
        # Evaluate(
        #     lambda G: model.predict_and_decode(
        #         G, complement=True, thresholdloopnodes=thresholdloopnodes
        #     ),
        #     f"{model_name}_decode_comp_true_k_10",
        # ),
        # Evaluate(
        #     lambda G: model.predict_and_decode(
        #         G,
        #         num_walkers=20,
        #         complement=True,
        #         thresholdloopnodes=thresholdloopnodes,
        #     ),
        #     f"decode_comp_true_k_20",
        # ),
        Evaluate(
            lambda G: model.least_probable_removal(G, compute_once=True),
            f"ldr_once",
            should_normalize_by_max_clique_size=False,
        ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G, compute_once=True, reversed_order=True,
        #     ),
        #     f"ldr_reverse_once",
        #     should_normalize_by_max_clique_size=False,
        # ),
        Evaluate(
            lambda G: model.least_probable_removal(
                G,
            ),
            f"ldr",
            should_normalize_by_max_clique_size=False,
        ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G,
        #         use_1st_pc=True,
        #     ),
        #     f"ldr_pca",
        # ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(G, reversed_order=True),
        #     f"ldr_reverse",
        #     should_normalize_by_max_clique_size=False,
        # ),
    ]


model_paths = [
    # "com-youtube_original", "com-orkut_original",
    # "imdb_binary_original", "collab_original",
    #     "n500_easy_original", "n500_medium_original", "n500_hard_original",
    #     "n1000_easy_instances_15_epochs_wc", "n1000_hard_instances_15_epochs_wc",
    #      "twitter_original",
    #     "n1000_medium_instances_15_epochs_wc",
    "n1000_medium_original"
]


def main():
    for mi, model_name in enumerate(model_paths):
        model_prefix = f"../models/erdos/diameter/{model_name}.pth"
        output_dir = "../results_attr"
        evals = [*generate_models_evaluates(model_prefix, model_name, device="cpu")]

        for n in [1000]:
            for theta in [1]:
                for k in range(90, n, 1):
                    instances = []
                    output_path = os.path.join(output_dir, f"n_{n}_k_{k}_beta_{theta}.jsonl")
                    for _ in range(100):
                        print(f"generated {_+1} out of 100")
                        X, S, Sigma, v = generate_sample_covariance(
                            n, n, k, theta, None
                        )
                        true_idx = np.flatnonzero(v)

                        S = np.abs(S)
                        instance = {
                            "G": (k, S, v, theta),
                            "v": v,
                            "max_clique_size": k,
                        }
                        instances.append(instance)


                    calculated_rows = get_calculated_rows(output_path, unique=True, cached=False)
                    print(f"calculated_rows: {len(instances)}")
                    for i, row in enumerate(instances):
                        if i in calculated_rows:
                            print(f"Skipping row {i} already calculated")
                            continue
                        handle_row(output_path, row, i, evals)



if __name__ == "__main__":
    main()

