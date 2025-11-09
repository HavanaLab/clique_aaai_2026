import json

import pandas as pd
from evaluate import Evaluate, EvaluateDataset, evaluate_graph_func, get_calculated_rows, handle_row
from graphs.graph_utils_numpy import (
    get_degrees,
    least_probable_removal, greedy_clique, aks,
)
import os
import numpy as np

from utils.pca import generate_sample_covariance


def main():
    output_base = "results/diameter/sprase_pca"
    output_path = os.path.join(output_base, "sparse_pca_eval_results.jsonl")
    os.makedirs(output_base, exist_ok=True)
    evals = [
        Evaluate(
            # lambda G: len(least_probable_removal(G, get_degrees)[0]), f"ldr_degrees"
            lambda G: least_probable_removal(G, get_degrees, compute_once=False)[1][-1], f"ldr_degrees",
            should_normalize_by_max_clique_size=False
        ),
        Evaluate(
            # lambda G: len(least_probable_removal(G, get_degrees)[0]), f"ldr_degrees"
            lambda G: least_probable_removal(G, get_degrees, compute_once=True)[1][-1], f"topk_degrees",
            should_normalize_by_max_clique_size=False
        ),
        # Evaluate(
        #     # lambda G: len(least_probable_removal(G, get_degrees)[0]), f"ldr_degrees"
        #     lambda G: aks(G), f"aks_degrees",
        #     should_normalize_by_max_clique_size=True
        # ),
        Evaluate(
            # lambda G: len(least_probable_removal(G, get_degrees)[0]), f"ldr_degrees"
            lambda G: greedy_clique(G[1]), f"degree_degrees",
            should_normalize_by_max_clique_size=True
        ),
        Evaluate(
            # lambda G: len(least_probable_removal(G, get_degrees)[0]), f"ldr_degrees"
            lambda G: G[0], f"k",
            should_normalize_by_max_clique_size=False
        ),
    ]
    instances = []
    for n in [1000]:
        for theta in [1]:
            for k in range(1, 51, 1):
                for _ in range(100):
                    X, S, Sigma, v = generate_sample_covariance(n, n, k, theta, None)
                    true_idx = np.flatnonzero(v)
                    S = np.abs(S)
                    instance = {
                        "G": (k, np.abs(S), v, theta),
                        "v": v,
                        "max_clique_size": k,
                    }
                    instances.append(instance)
                    np.abs(S)

    calculated_rows = get_calculated_rows(output_path, unique=True, cached=False)
    print(f"calculated_rows: {len(calculated_rows)}")
    for i, row in enumerate(instances):
        if i in calculated_rows:
            print(f"Skipping row {i} already calculated")
            continue
        handle_row(output_path, row, i, evals)





def process_directory(directory, key="ldr_degrees"):
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
    main()

    # Example usage
    directory_path = "results/diameter/sprase_pca"
    process_directory(directory_path)