import json

import pandas as pd

from evaluate import Evaluate, EvaluateDataset, evaluate_graph_func, get_calculated_rows
from erdos.erdos_model import ErdosModel


from prefect import task, flow
from prefect_ray.task_runners import RayTaskRunner





def generate_models_evaluates(model_path, model_name, device="cuda"):
    model = ErdosModel(model_path, device=device)
    model.model.eval()
    return [
        # Evaluate(
        #     lambda G: model.predict_and_decode(G, num_walkers=20),
        #     f"decode_k_20",
        # ),
        Evaluate(
            lambda G: model.least_probable_removal(
                G, compute_once=True, should_remove_isolated_nodes=True
            ),
            f"LDR_ONCE",
        ),
        Evaluate(
            lambda G: model.least_probable_removal(
                G, should_remove_isolated_nodes=True
            ),
            f"LDR",
        ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G,
        #         should_remove_isolated_nodes=True,
        #         use_1st_pc=True,
        #         use_1st_pc_layer=3,  # The last convolution layer
        #         fix_by_spearman=True,
        #     ),
        #     f"LDR_PC1",
        # ),
        # Evaluate(
        #     lambda G: model.least_probable_removal(
        #         G,
        #         should_remove_isolated_nodes=True,
        #         use_1st_pc=True,
        #         use_1st_pc_layer=3,  # The last convolution layer
        #         fix_by_spearman=True,
        #         compute_once=True,
        #     ),
        #     f"LDR_PC1_once",
        # ),
    ]


@task(retries=99999, retry_delay_seconds=10, retry_jitter_factor=0.5, log_prints=False)
def handle_row(output_path, row, row_num, evals):
    print(f"======{row_num}======")
    result = evaluate_graph_func(row, evals)
    result["row_num"] = row_num
    with open(output_path, "a", 1) as f:
        json.dump(result, f)
        f.write("\n")


model_paths = [
    "com-youtube_original",
    "com-orkut_original",
    "imdb_binary_original",
    "collab_original",
    "n500_easy_original",
    "n500_medium_original",
    "n500_hard_original",
    "n1000_easy_original",
    "n1000_medium_original",
    "n1000_hard_original",
    "twitter_original",
]

datasets_paths = [
    "collab",
    "imdb_binary",
    "twitter",
    "com-youtube",
    "com-orkut",
    "facebook",
    "n1000_hard_instances",
    "n1000_medium_instances",
    "n1000_easy_instances",
    "n500_easy_instances",
    "n500_medium_instances",
    "n500_hard_instances",
]


@flow(
    log_prints=False,
    task_runner=RayTaskRunner(
        init_kwargs=dict(
            include_dashboard=False,
        )
    ),
)
def main():
    for di, dataset_name in enumerate(datasets_paths):
        for mi, model_name in enumerate(model_paths):
            dataset_path = f"datasets/{dataset_name}.jsonl"
            model_prefix = f"models/erdos/{model_name}.pth"
            output_path = f"results/erdos approximation ratio/{model_name}_on_{dataset_name}_eval_results.jsonl"

            evals = [
                *generate_models_evaluates(model_prefix, model_name, device="cpu"),
            ]
            eval_dataset = EvaluateDataset(evals, input_path=dataset_path, output_path=output_path)

            calculated_rows = get_calculated_rows(output_path, unique=True, cached=False)
            print(f"calculated_rows: {len(calculated_rows)}")
            submitted = []
            for i, row in enumerate(eval_dataset.dataset):
                if i in calculated_rows:
                    print(f"Skipping row {i} already calculated")
                    continue

                row_future = handle_row.with_options(
                    name=f"model:{model_name} dataset:{dataset_name} row_num:{i}",
                    tags=[
                        "twenty",
                        f"dataset:{dataset_name}",
                        f"model:{model_name}",
                    ],
                ).submit(output_path, row, i, evals)
                submitted.append(row_future)
                if len(submitted) >= 96:
                    for future in submitted:
                        future.wait()
                    submitted = []


if __name__ == "__main__":
    main()
