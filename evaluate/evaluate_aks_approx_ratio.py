import json
import numbers

import pandas as pd
from evaluate import Evaluate, EvaluateDataset, get_calculated_rows, handle_row
from graphs.graph_utils_numpy import (
    get_degrees,
    least_probable_removal,
)
import os
import numpy as np
import networkx as nx
from evaluate import Evaluate, EvaluateDataset
from graphs.graph_utils_numpy import  aks, is_clique
from graphs.clique_solver import get_max_clique


def evaluate_graph_func(instance, evals):
    G = instance["G"]
    max_clique_size = instance["max_clique_size"]
    result_row = {}
    for eval in evals:
        print(f"evaluating {eval.name}")
        ev_result = aks(G, max_clique_size)
        mask = np.zeros(G.shape[0], dtype=bool)
        mask[ev_result] = True
        if not is_clique(G[:, mask][mask, :]):
            # run pmc on the subset graph induced by ev_result
            subset_G = G[np.ix_(ev_result, ev_result)]
            # turn to nx graph
            subset_G = nx.from_numpy_array(subset_G)
            ev_result = get_max_clique(subset_G, to_root="../")

        if not isinstance(ev_result, numbers.Number):
            ev_result = len(ev_result)
        if eval.should_normalize_by_max_clique_size:
            result_row[eval.name] = ev_result / max_clique_size
        print(f"{eval.name}: {result_row[eval.name]}")
        result_row["max_clique_size"] = max_clique_size
    return result_row


datasets_paths = [
    "n500_easy_instances",
    "n500_medium_instances",
    "n500_hard_instances",
    "n1000_easy_instances",
    "n1000_medium_instances",
    "n1000_hard_instances",
    "collab",
    "imdb_binary",
    "twitter",
    "com-youtube",
    "com-orkut",
    "facebook",
]

model_paths = [
    "aks",
]


def main():
    for di, dataset_name in enumerate(datasets_paths):
        for mi, model_name in enumerate(model_paths):
            dataset_path = f"../datasets/{dataset_name}.jsonl"
            output_base = f"../results/aks approximation ratio/"
            output_path = os.path.join(output_base, f"ldr_on_{dataset_name}_eval_results.jsonl")
            os.makedirs(output_base, exist_ok=True)

            evals = [Evaluate(aks, f"aks")]
            eval_dataset = EvaluateDataset(evals, input_path=dataset_path, output_path=output_path)

            calculated_rows = get_calculated_rows(output_path, unique=True, cached=False)
            print(f"calculated_rows: {len(calculated_rows)}")
            for i, row in enumerate(eval_dataset.dataset):
                if i in calculated_rows:
                    print(f"Skipping row {i} already calculated")
                    continue
                handle_row(output_path, row, i, evals, evaluate_graph_func_=evaluate_graph_func)


def process_directory(directory, key="aks"):
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
    directory_path = "../results/aks approximation ratio"
    process_directory(directory_path, key="aks")