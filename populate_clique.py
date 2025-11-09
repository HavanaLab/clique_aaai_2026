import json

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from generate_difficult_instances import load_and_parse_from_jsonl, save_to_json, turn_instance_to_dict
from pmc.pmc import pmc

dataset_names = [
    "collab.jsonl",
    "com-orkut.jsonl",
    "com-youtube.jsonl",
    "facebook.jsonl",
    "imdb_binary.jsonl",
    "twitter.jsonl",
    "n500_easy_instances.jsonl",
    "n500_hard_instances.jsonl",
    "n500_medium_instances.jsonl",
    "n1000_easy_instances.jsonl",
    "n1000_hard_instances.jsonl",
    "n1000_medium_instances.jsonl",
]
def get_max_clique(G):
    adjacency_matrix = np.array(G)
    ei, ej = np.where(adjacency_matrix == 1)
    number_of_nodes = adjacency_matrix.shape[0]
    number_of_edges = len(ei)
    max_clique = pmc(ei.tolist(), ej.tolist(), number_of_nodes, number_of_edges)
    return max_clique

if __name__ == "__main__":
    for dataset_name in dataset_names:
        clique_size = []
        dataset_path = f"datasets/{dataset_name}"
        output_dataset_path = f"datasets/{dataset_name}_populated_clique.jsonl"
        with open(dataset_path, "r") as file:
            for line in file:
                data = json.loads(line)
                graph = np.array(data["adjacency_matrix"])
                clique = get_max_clique(graph)
                if "clique" not in data:
                    data["clique"] = clique.tolist()
                else:
                    print("had to use clique_ and not clique")
                    data["clique_"] = clique.tolist()
                save_to_json(output_dataset_path, [data], lines=True, write_mode="a")
