import numpy as np
import matplotlib.pyplot as plt
import os

from generate_difficult_instances import load_and_parse_from_jsonl

dataset_names = [
    "collab.jsonl",
    "com-orkut.jsonl",
    "com-youtube.jsonl",
    "facebook.jsonl",
    "imdb_binary.jsonl",
    "twitter.jsonl",
    "n1000_easy_instances.jsonl",
    "n1000_hard_instances.jsonl",
    "n1000_medium_instances.jsonl",
    "n500_easy_instances.jsonl",
    "n500_hard_instances.jsonl",
    "n500_medium_instances.jsonl",
]


def plot_degree_distribution(degrees, title="Degree Distribution"):
    """
    Plot the histogram of node degrees.

    Args:
      degrees (array-like): the degree of each node.
      title (str): title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, edgecolor="black", rwidth=0.8)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()



if __name__ == "__main__":
    if not os.path.exists("results/clique_size_dist_norm"):
        os.makedirs("results/clique_size_dist_norm")

    for dataset_name in dataset_names:
        clique_size = []
        dataset_path = f"datasets/{dataset_name}"
        dataset = load_and_parse_from_jsonl(dataset_path, to_networkx=False)
        for i, instance in enumerate(dataset):
            graph = instance["G"]
            max_clique_size_ = instance["max_clique_size"]
            if type(max_clique_size_) != int:
                max_clique_size_= len(max_clique_size_)

            max_clique_size_ = max_clique_size_ / graph.shape[0]
            clique_size.append(max_clique_size_)

        flattened_clique_size = np.array(clique_size)
        np.save(f"results/clique_size_dist_norm/{dataset_name}_clique_size.npy", flattened_clique_size)

    # # Example to load and plot
    for dataset_name in dataset_names:
        flattened_degrees = np.load(f"results/clique_size_dist_norm/{dataset_name}_clique_size.npy")
        plot_degree_distribution(flattened_degrees, f"agg degree from dataset {dataset_name},"
                                                    f"min: {round(np.min(flattened_degrees), 3)}, "
                                                    f"max: {round(np.max(flattened_degrees), 3)}, "
                                                    f"mean: {round(np.mean(flattened_degrees), 3)}, "
                                                    f"std: {round(np.std(flattened_degrees), 3)}, "
                                 )
