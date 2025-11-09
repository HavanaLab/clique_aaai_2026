import random
import json
import os
import numpy as np
from tqdm import tqdm

from utils.pca import generate_sample_covariance

if __name__ == "__main__":
    n = 1000  # number of samples
    p = 1000   # number of features
    theta = 1  # spike strength
    data = []
    output_file = os.path.join("datasets", "sparse_pca.jsonl")
    with open(output_file, "w") as f:
        for _ in tqdm(range(1000)):
            k = random.randint(1, 21)  # sparsity level
            x, s, sigma, v = generate_sample_covariance(n ,n, k,  theta, None)
            row = {
                "adjacency_matrix": s.tolist(),
                "max_clique_size": k,
                "clique": v.tolist(),
                         }
            json.dump(row, f)
            f.write("\n")
