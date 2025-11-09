import json
import os.path
from functools import cache
import numbers

import pandas as pd
import numpy as np
from evaluate import Evaluate, EvaluateDataset, get_calculated_rows, handle_row, \
    get_dataset_length  # , evaluate_graph_func
from model import ScatteringNoFeaturesModel




def generate_models_evaluates(
    model_path, model_name, device="cuda", thresholdloopnodes=300
):
    model = ScatteringNoFeaturesModel(model_path, device=device, use_networkx=False)
    return [
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
        #     f"{model_name}_decode_k_20",
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
        #     f"{model_name}_decode_comp_true_k_20",
        # ),
        Evaluate(
            lambda G: model.least_probable_removal(G, compute_once=True),
            f"ldr_once",
        ),
        Evaluate(
            lambda G: model.least_probable_removal(
                G, compute_once=True, reversed_order=True
            ),
            f"ldr_reverse_once",
        ),
        Evaluate(
            lambda G: model.least_probable_removal(
                G,
            ),
            f"ldr",
        ),
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
        Evaluate(
            lambda G: model.least_probable_removal(G, reversed_order=True),
            f"ldr_reverse",
        ),
    ]


import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # print(f"[{func.__name__}] executed in {end - start:.4f} seconds")
        return result, end - start
        # return result

    return wrapper


def evaluate_graph_func(instance, evals):
    G = instance["G"]
    max_clique_size = instance["max_clique_size"]
    result_row = {}
    for eval in evals:
        print(f"evaluating {eval.name}")
        timed_eval = timeit(eval)
        # ev_result = eval(G)
        ev_result, eval_time = timed_eval(G)
        if not isinstance(ev_result, numbers.Number):
            ev_result = len(ev_result)
        if eval.should_normalize_by_max_clique_size:
            result_row[eval.name] = ev_result / max_clique_size
        result_row[f"{eval.name}_time"] = eval_time
        print(f"{eval.name}: {result_row[eval.name]}")
        print(f"{eval.name}_time: {eval_time}")
    return result_row



datasets_paths = [
    "collab",
    "imdb_binary",
    "twitter",
    "com-youtube",
    "com-orkut",
    "facebook",
    "n1000_easy_instances",
    "n1000_medium_instances",
    "n1000_hard_instances",
    "n500_easy_instances",
    "n500_medium_instances",
    "n500_hard_instances",
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
    "n1000_medium_instances_15_epochs_wc",
]



def main():
    for di, dataset_name in enumerate(datasets_paths):
        for mi, model_name in enumerate(model_paths):
            dataset_path = f"../datasets/{dataset_name}.jsonl"
            model_prefix = f"../models/scattering/{model_name}.pth"

            # output_path = f"results/scattering no features approximation ratio/{model_name}_on_{dataset_name}_eval_results.jsonl"
            output_basepath = "../results/run time/scattering no features approximation ratio/"
            output_path = os.path.join(output_basepath, f"{model_name}_on_{dataset_name}_eval_results.jsonl")
            os.makedirs(output_basepath, exist_ok=True)

            evals = [*generate_models_evaluates(model_prefix, model_name, device="cpu")]
            eval_dataset = EvaluateDataset(evals, input_path=dataset_path, output_path=output_path)
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
                handle_row(output_path, row, i, evals, evaluate_graph_func_=evaluate_graph_func)


if __name__ == "__main__":
    print("started")
    main()
