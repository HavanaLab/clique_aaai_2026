import json
from functools import cache
from typing import List
import numbers

import pandas as pd

from generate_difficult_instances import load_and_parse_from_json
from generate_difficult_instances import load_and_parse_from_jsonl

@cache
def get_dataset_length(dataset_path):
    try:
        print("in get dataset length")
        dataset = load_and_parse_from_jsonl(dataset_path, to_networkx=False)
        print("done get dataset length")
        return sum(1 for _ in dataset)
    except:
        print("couldnt read dataset")
        return 0

@cache
def get_calculation_rows_cached(output_path, unique=False):
    return get_calculation_rows_raw(output_path, unique=unique)

def get_calculation_rows_raw(output_path, unique=False):
    try:
        df = pd.read_json(output_path, lines=True)
        if unique is False:
            return list(df.index)
        else:
            return df["row_num"].unique().tolist()
    except:
        return []

def get_calculated_rows(output_path, unique=False, cached=False):
    if cached:
        return get_calculation_rows_cached(output_path, unique=unique)
    else:
        return get_calculation_rows_raw(output_path, unique=unique)


def evaluate_graph_func(instance, evals):
    G = instance["G"]
    max_clique_size = instance["max_clique_size"]
    # G = (G, max_clique_size)
    result_row = {}
    for eval in evals:
        print(f"evaluating {eval.name}")
        ev_result = eval(G)
        if not isinstance(ev_result, numbers.Number):
            ev_result = len(ev_result)
        if eval.should_normalize_by_max_clique_size:
            result_row[eval.name] = ev_result / max_clique_size
        else:
            result_row[eval.name] = ev_result
        print(f"{eval.name}: {result_row[eval.name]}")

    return result_row


def handle_row(output_path, row, row_num, evals, evaluate_graph_func_=evaluate_graph_func):
    print(f"======{row_num}======")
    result = evaluate_graph_func_(row, evals)
    result["row_num"] = row_num
    with open(output_path, "a", 1) as f:
        json.dump(result, f)
        f.write("\n")


class Evaluate:
    def __init__(self, func, name: str, should_normalize_by_max_clique_size=True):
        self.func = func
        self.name = name
        self.should_normalize_by_max_clique_size = should_normalize_by_max_clique_size

    def __call__(self, G):
        return self.func(G)


class EvaluateDataset:
    def __init__(
        self,
        evals: List[Evaluate],
        input_path,
        output_path,
        lines=True,
        to_networkx=False,
    ) -> None:
        self.evals = evals
        self.output_path = output_path
        self.input_path = input_path
        self.dataset = load_and_parse_from_json(
            self.input_path, lines=lines, to_networkx=to_networkx
        )

    def evaluate_graph(self, instance):
        G = instance["G"]
        max_clique_size = instance["max_clique_size"]
        result_row = {}
        for eval in self.evals:
            print(f"evaluating {eval.name}")
            ev_result = eval(G)
            # if isinstance(ev_result, set) or isinstance(ev_result, list):
            if not isinstance(ev_result, numbers.Number):
                ev_result = len(ev_result)
            result_row[eval.name] = ev_result / max_clique_size
        return result_row

    def run(self, write_mode="a", buffer_size=1, start=0):
        with open(self.output_path, write_mode, buffer_size) as f:
            for i, row in enumerate(self.dataset):
                if i < start:
                    continue
                print(f"======{i}======")
                result = self.evaluate_graph(row)
                json.dump(result, f)
                f.write("\n")
        print("finished saving dataset")
