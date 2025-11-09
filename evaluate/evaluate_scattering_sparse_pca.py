import json
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from evaluate import Evaluate, EvaluateDataset, evaluate_graph_func, get_calculated_rows, handle_row
from graphs.graph_utils_numpy import get_degrees
from model import ScatteringNoFeaturesModel

from utils.pca import generate_sample_covariance, covariance_thresholding_grid_search, run_sklearn_sparse_pca, covariance_thresholding_S
from sklearn.decomposition import SparsePCA




def generate_models_evaluates(
    model_path, model_name, device="cuda", thresholdloopnodes=300
):
    model = ScatteringNoFeaturesModel(model_path, device=device, use_networkx=False)
    # model = ErdosModel(model_path, device=device)
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
    "n1000_medium_instances_15_epochs_wc",
    # "n1000_medium_original"
]


def main():

    for mi, model_name in enumerate(model_paths):
        model_prefix = f"../models/no_features/{model_name}.pth"
        output_base = "../results/pca_recall/scattering"
        os.makedirs(output_base, exist_ok=True)

        evals = [*generate_models_evaluates(model_prefix, model_name, device="cpu")]
        for n in [1000]:
            for theta in [1]:
                for k in range( 1, n, 1):
                    instances = []
                    output_path = os.path.join(output_base, f"n_{n}_k_{k}_beta_{theta}.jsonl")
                    for _ in range(100):
                        X, S, Sigma, v = generate_sample_covariance(n,n, k,  theta, None)
                        true_idx = np.flatnonzero(v)
                        c, max_ratio = covariance_thresholding_grid_search(S, k, true_idx)

                        str_to_print = "iter: {}, n: {}, k: {}, c:{}, ratio: {}, optimal: {}".format(
                            _, n,k,c,max_ratio,
                            -1 #                                    len(set(true_idx).intersection(set(optimal_selected_indices)))/len(true_idx)
                            )
                        with open(os.path.join(output_base,"conv_thresholding.txt"), "a") as f:
                            f.write(str_to_print)
                            f.write("\n")
                        print(str_to_print)

                        S = np.abs(S)
                        instance = {
                            "G": (k, S, v, theta),
                            "v": v,
                            "max_clique_size": k,
                        }
                        instances.append(instance)

                    # continue
                    calculated_rows = get_calculated_rows(output_path, unique=True, cached=False)
                    print(f"calculated_rows: {len(instances)}")
                    for i, row in enumerate(instances):
                        if i in calculated_rows:
                            print(f"Skipping row {i} already calculated")
                            continue
                        handle_row(output_path, row, i, evals)


if __name__ == "__main__":
    main()


    # def process_directory(directory):
    #     # List all files in the directory
    #     for file_name in os.listdir(directory):
    #         file_path = os.path.join(directory, file_name)
    #
    #         # Check if it's a file
    #         if os.path.isfile(file_path):
    #             ldr_values = []
    #             topk = []
    #             # Split file name by "_" and get the second index
    #             # parts = file_name.split("_")
    #             # if len(parts) > 2:
    #             #     index_2 = parts[2]
    #             #     if index_2[0] == "n":
    #             #         if "1000" not in index_2:
    #             #             continue
    #             #         index_2 += "_" + parts[3]
    #             try:
    #                 with open(file_path, 'r') as f:
    #                     for line in f:
    #                         data = json.loads(line)
    #                         topk.append(max(data["ldr_once"], data["ldr_reverse_once"]))
    #                         ldr_values.append(max(data["ldr"], data["ldr_reverse"]))
    #             except Exception as e:
    #                 print(f"Error processing {file_name}: {e}")
    #
    #         # Calculate average and standard deviation
    #         if ldr_values:
    #             average = np.mean(ldr_values)
    #             std_dev = np.std(ldr_values)
    #             print(file_name, f"Average: {round(average,2)} \pm {round(std_dev,2)}   AND topk: {round(np.mean(topk),2)} \pm {round(np.std(topk),2)}")
    #         else:
    #             print("No 'ldr_degrees' values found.")
    #
    # # Example usage
    # directory_path = "/home/elad/Documents/clique/CliqueEvaluation/results/scattering no features approximation ratio"
    # process_directory(directory_path)