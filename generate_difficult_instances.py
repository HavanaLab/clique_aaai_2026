import json
from typing import Literal, Union
import networkx as nx
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

from graphs.graph_utils_numpy import is_clique

from graphs.graph_utils_numpy import least_degree_removal


@dataclass
class Args:
    n: int
    k: int
    p: float
    number_of_graphs: int
    output_file: str
    difficulty: Union[Literal["Easy"] | Literal["Hard"] | Literal["Medium"] | Literal["None"] | Literal["Intermediate"]] = "Hard"
    plant: bool = False


def is_clique_nx(G):
    n = len(G.nodes)
    m = len(G.edges)
    return m == n * (n - 1) / 2


def unify_complete_and_uniform_graphs(k, n, p=0.5):
    # Create a complete graph of size k
    complete_graph = nx.complete_graph(k)
    random_graph = nx.gnp_random_graph(n, p)
    return nx.compose(complete_graph, random_graph)


def top_degree_clique(G, k):
    degrees = G.degree()
    degrees = {v: d for v, d in degrees}
    sorted_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_k_degrees = sorted_by_degree[:k]
    vertices_top_k = [v for v, d in top_k_degrees]
    k_t = set(vertices_top_k)
    subgraph = G.subgraph(k_t)

    k_t_is_clique = is_clique(nx.to_numpy_array(subgraph))
    #k_t_is_clique = is_clique_nx(subgraph) # alternativly, use is_clique_nx

    return k_t_is_clique


def generate(args: Args, ensure_difficulty=False):
    satisfies_difficulty = False
    while not satisfies_difficulty:
        G = unify_complete_and_uniform_graphs(args.k, args.n, args.p)

        max_clique_size = args.k
        easy_indicator = top_degree_clique(G, max_clique_size)
        G_np = nx.to_numpy_array(G)
        medium_indicator = (
            len(least_degree_removal(G_np)) == max_clique_size and not easy_indicator
        )
        hard_indicater = not easy_indicator and not medium_indicator
        match args.difficulty:
            case "Easy":
                satisfies_difficulty = easy_indicator
            case "Medium":
                satisfies_difficulty = medium_indicator
            case "Hard":
                satisfies_difficulty = hard_indicater
            case _:
                ensure_difficulty = False
                print("No known difficulty specified, accepting any instance. Type is: ", _)
                print("Overriding ensure_difficulty to False.")
        if not ensure_difficulty:
            break
    return G, max_clique_size, easy_indicator, medium_indicator, hard_indicater


def turn_instance_to_dict(
    G, max_clique_size, easy_indicator, medium_indicator, hard_indicater, args: Args, clique=None
):
    return {
        "adjacency_matrix": nx.adjacency_matrix(G).todense().tolist(),
        "max_clique_size": max_clique_size,
        "easy_indicator": easy_indicator,
        "medium_indicator": medium_indicator,
        "hard_indicater": hard_indicater,
        "clique": clique,
        "generation_args": args.__dict__,
    }


def load_and_parse_from_jsonl(
    file: str,
    to_networkx=True,
    permute=False,
    graph_key="adjacency_matrix",
    load_indicators=False,
):
    with open(file, "r") as f:
        for line_i, line in enumerate(f):
            data = json.loads(line)
            if isinstance(data, str):
                data = json.loads(data)
            G = np.array(data[graph_key])
            # permute rows and columns
            if permute:
                perms = np.random.permutation(G.shape[0])
                G = G[perms, :]
                G = G[:, perms]
            if to_networkx:
                sparse_similarity_matrix = csr_matrix(G)
                G = nx.from_scipy_sparse_array(sparse_similarity_matrix)
                G = G.to_undirected()
            max_clique_size = data["max_clique_size"]
            clique = data["clique"] if "clique" in data else None
            if not load_indicators:
                yield dict(G=G, clique=clique, max_clique_size=max_clique_size)
                continue
            easy_indicator = data["easy_indicator"]
            medium_indicator = data["medium_indicator"]
            hard_indicater = data["hard_indicater"]
            if "args" in data.keys():
                args = Args(**data["args"])
            elif "generation_args" in data.keys():
                args = Args(**data["generation_args"])
            else:
                args = {}
            yield dict(
                G=G,
                max_clique_size=max_clique_size,
                clique=clique,
                easy_indicator=easy_indicator,
                medium_indicator=medium_indicator,
                hard_indicater=hard_indicater,
                args=args,
            )


def load_and_parse_from_json(
    file: str, lines=False, to_networkx=True, graph_key="adjacency_matrix"#, skip=0,
):
    if lines:
        return load_and_parse_from_jsonl(
            file, to_networkx=to_networkx, graph_key=graph_key, #skip=skip,
        )
    else:
        with open(file, "r") as f:
            dataset = json.load(f)
        parsed_dataset = []
        for data in dataset:
            G = np.array(data[graph_key])
            if to_networkx:
                G = nx.from_numpy_array(G)
                G = G.to_undirected()

            max_clique_size = data["max_clique_size"]
            easy_indicator = data["easy_indicator"]
            medium_indicator = data["medium_indicator"]
            hard_indicater = data["hard_indicater"]
            if "args" in data.keys():
                args = Args(**data["args"])
            else:
                args = Args(**data["generation_args"])
            # append dictionary with same  keys with parsed graph
            parsed_dataset.append(
                dict(
                    G=G,
                    max_clique_size=max_clique_size,
                    easy_indicator=easy_indicator,
                    medium_indicator=medium_indicator,
                    hard_indicater=hard_indicater,
                    args=args,
                )
            )

        return parsed_dataset


def save_to_json(file: str, dataset, lines=False, write_mode="w"):
    with open(file, write_mode) as f:
        if lines:
            for row in dataset:
                json.dump(row, f)
                f.write("\n")
        else:
            json.dump(dataset, f)


create = [
    (71, 500, Args(n=500,output_file="n500_easy_instances.jsonl",difficulty="Easy",k=-1,p=0.5,number_of_graphs=1000,plant=True)),
    (26, 70, Args(n=500,output_file="n500_medium_instances.jsonl",difficulty="Medium",k=-1,p=0.5,number_of_graphs=1000,plant=True)),
    (15, 25, Args(n=500,output_file="n500_hard_instances.jsonl",difficulty="Hard",k=-1,p=0.5,number_of_graphs=1000,plant=True)),

    (62, 1000, Args(n=1000, output_file="n1000_easy_instances.jsonl", difficulty="Easy", k=-1, p=0.5, number_of_graphs=1000, plant=True)),
    (36, 61, Args(n=1000, output_file="n1000_medium_instances.jsonl", difficulty="Medium", k=-1, p=0.5, number_of_graphs=1000, plant=True)),
    (20, 35, Args(n=1000, output_file="n1000_hard_instances.jsonl", difficulty="Hard", k=-1, p=0.5, number_of_graphs=1000, plant=True)),
]


if __name__ == "__main__":
    for k_lower, k_upper, args in create:
        for _ in tqdm(range(args.number_of_graphs)):
            k = np.random.randint(k_lower, k_upper + 1)
            args.k = k
            instance = generate(args)
            instance_dict = turn_instance_to_dict(*instance, args)
            save_to_json(args.output_file, [instance_dict], lines=True, write_mode="a")
