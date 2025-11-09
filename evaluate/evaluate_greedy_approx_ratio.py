import json

import pandas as pd
from evaluate import Evaluate, EvaluateDataset, evaluate_graph_func, get_calculated_rows, handle_row

from graphs.graph_utils_numpy import greedy_clique
import os
import numpy as np



datasets_paths = [
    # "../datasets/collab.jsonl",
    # "../datasets/com-orkut.jsonl",
    # "../datasets/com-youtube.jsonl",
    # "../datasets/facebook.jsonl",
    # "../datasets/imdb_binary.jsonl",
    # "../datasets/twitter.jsonl",
    # "../datasets/n1000_easy_instances.jsonl",
    # "../datasets/n1000_gnp_instances.jsonl",
    # "../datasets/n1000_hard_instances.jsonl",
    # "../datasets/n1000_medium_instances.jsonl",
    # "../datasets/n500_easy_instances.jsonl",
    # "../datasets/n500_hard_instances.jsonl",
    "../datasets/n500_medium_instances.jsonl",
]


def main():
    for di, dataset_name in enumerate(datasets_paths):
        print("starting dataset", dataset_name)
        # dataset_path = f"datasets/{dataset_name}.jsonl"
        dataset_path = dataset_name
        dataset_name = dataset_name.split("/")[-1].split(".")[0]
        output_base = "../results/diameter/greedy/"
        output_path = os.path.join(output_base,  f"{dataset_name}.jsonl")
        os.makedirs(output_base, exist_ok=True)

        evals = [Evaluate(greedy_clique, f"greedy")]
        eval_dataset = EvaluateDataset(evals, input_path=dataset_path, output_path=output_path)

        calculated_rows = get_calculated_rows(output_path, unique=True, cached=False)
        for i, row in enumerate(eval_dataset.dataset):
            if i in calculated_rows:
                print(f"Skipping row {i} already calculated")
                continue
            handle_row(output_path, row, i, evals)

def process_directory(directory, key="greedy"):
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
    directory_path = "../results/diameter/greedy"
    process_directory(directory_path)