import json
import os
from itertools import combinations
from functools import cache
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from evaluate import Evaluate, EvaluateDataset, evaluate_graph_func, get_calculated_rows, handle_row, get_dataset_length
from evaluate_erdos_spearmen_correlation_and_pca import compute_spearman
from graphs.graph_utils_numpy import get_degrees
from model import ScatteringNoFeaturesModel
from utils.pca import generate_sample_covariance


def generate_models_evaluates(
    model_path, model_name, device="cuda"
):
    model = ScatteringNoFeaturesModel(model_path, device=device, use_networkx=False)
    return [
        Evaluate(
            lambda G: model.least_probable_removal(G, compute_once=True ),
            f"ldr_once",
        ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G, compute_once=True, reversed_order=True
        #     ),
        #     f"ldr_reverse_once",
        # ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G,
        #     ),
        #     f"ldr",
        # ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G,
        #         use_1st_pc=True,
        #     ),
        #     f"ldr_pca",
        # ),
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
        #     lambda G: model.least_probable_removal(G, reversed_order=True),
        #     f"ldr_reverse",
        # ),
    ]


datasets_paths = [
    # "com-youtube",
    # "facebook",
    # "com-orkut",
    # "imdb_binary",
    # "collab",
    # "n500_easy_instances",
    # "n500_medium_instances",
    # "n500_hard_instances",
    # "n1000_easy_instances",
    "n1000_medium_instances",
    # "n1000_hard_instances",
    # "twitter",

]

model_paths = [
    # "com-youtube_original",
    # "com-orkut_original",
    # "imdb_binary_original",
    # "collab_original",
    # "n500_easy_original",
    # "n500_medium_original",
    # "n500_hard_original",
    # "n1000_easy_instances_15_epochs_wc",
    # "n1000_hard_instances_15_epochs_wc",
    # "twitter_original",
    # "n1000_medium_instances_15_epochs_wc",
]


def main():
    for di, dataset_name in enumerate(datasets_paths):
        for mi, model_name in enumerate(model_paths):
            dataset_path = f"../datasets/{dataset_name}.jsonl"
            model_prefix = f"../models/scattering/{model_name}.pth"

            # output_path = f"results/scattering no features approximation ratio/{model_name}_on_{dataset_name}_eval_results.jsonl"
            output_base_dir = f"../results/scattering no features approximation ratio/results merged"
            output_path = os.path.join(output_base_dir,f"{model_name}_on_{dataset_name}_eval_results.jsonl")
            os.makedirs(output_base_dir, exist_ok=True)

            evals = [
                *generate_models_evaluates(model_prefix, model_name, device="cpu"),
            ]
            eval_dataset = EvaluateDataset( evals, input_path=dataset_path, output_path=output_path)

            ds_length = get_dataset_length(dataset_path)
            calculated_rows = get_calculated_rows(output_path, unique=True, cached=True)
            print(f"calculated_rows: {len(calculated_rows)}")

            if len(calculated_rows) == ds_length:
                print(f"Skipping m:{model_name} d:{dataset_name}")
                continue

            for i, row in enumerate(eval_dataset.dataset):
                if i in calculated_rows:
                    print(f"Skipping row {i} already calculated")
                    continue
                handle_row(output_path, row, i, evals)


def process_directory(directory, key="ldr_once"):
    # List all files in the directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        # Check if it's a file
        if not os.path.isfile(file_path):
            continue
        parts = file_name.split("_")
        if len(parts) <= 2:
            continue
        values = []
        data_set_name = create_name(parts)
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if key in data:
                        values.append(data[key])
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

        # Calculate average and standard deviation
        if values:
            average = np.mean(values)
            std_dev = np.std(values)
            print(data_set_name, f"Average: {round(average,2)} \pm {round(std_dev,2)}")
        else:
            print(f"No '{key}' values found.")


def create_name(parts):
    index_2 = parts[2]
    if index_2[0] == "n":
        index_2 += "_" + parts[3]
    return index_2

if __name__ == "__main__":
    print("started")
    main()

    # Example usage
    # directory_path = "/home/elad/Documents/clique/CliqueEvaluation/results/scattering no features approximation ratio"
    directory_path = "/home/elad/Documents/clique/CliqueEvaluation/results/scattering no features approximation ratio"
    process_directory(directory_path)