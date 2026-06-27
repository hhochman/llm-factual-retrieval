import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json

from tqdm import tqdm

from src.loaders import load_model
from src.utils import apply_path_and_check, collect_layer_inputs, set_seed
from src.experiments import run_sufficiency_test


def load_counterfact_samples(sample_path: str):
    with open(sample_path, "r") as f:
        samples = json.load(f)
    return {item["case_id"]: item for item in samples}


def load_path_results(path_results_path: str):
    results = {}
    with open(path_results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            results[entry["case_id"]] = entry
    return results


def model_dir_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def resolve_default_paths(model_name: str, path_mode: str):
    model_dir = model_dir_name(model_name)
    base_dir = os.path.join("data", model_dir)
    sample_path = os.path.join(base_dir, "counterfact_sample.json")
    path_results_path = os.path.join(base_dir, f"path_results_{path_mode}.jsonl")
    return sample_path, path_results_path


def main():
    parser = argparse.ArgumentParser(description="Run sufficiency experiments over prepared CounterFact samples.")
    parser.add_argument("--model", type=str, choices=["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B"], required=True)
    parser.add_argument("--path_mode", type=str, choices=["primary", "alternative"], default="primary")
    parser.add_argument(
        "--exp_type",
        type=str,
        choices=["representation_knockout", "downstream_injection", "path_continuation", "global_broadcast"],
        required=True,
    )
    parser.add_argument("--sample_path", type=str, default=None)
    parser.add_argument("--path_results_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    sample_path, path_results_path = resolve_default_paths(args.model, args.path_mode)
    if args.sample_path is not None:
        sample_path = args.sample_path
    if args.path_results_path is not None:
        path_results_path = args.path_results_path

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Loading sampled CounterFact data from: {sample_path}")
    samples_by_case_id = load_counterfact_samples(sample_path)

    print(f"Loading path results from: {path_results_path}")
    path_results_by_case_id = load_path_results(path_results_path)

    shared_case_ids = sorted(set(samples_by_case_id.keys()) & set(path_results_by_case_id.keys()))
    if not shared_case_ids:
        raise ValueError("No shared case_id values found between the sample file and the path results file.")

    total = 0
    failing = 0
    skipped = 0

    for case_id in tqdm(shared_case_ids, desc="Running sufficiency tests"):
        sample = samples_by_case_id[case_id]
        path_entry = path_results_by_case_id[case_id]

        entity = sample["entity"]
        counterfact_entity = sample["counterfact_entity"]
        prompt_template = sample["prompt"]
        target_token = sample["target_token"]
        ent_slice = slice(sample["entity_slice_start"], sample["entity_slice_stop"])

        clean_prompt = prompt_template.format(entity)
        counterfact_prompt = prompt_template.format(counterfact_entity)

        path_layers = path_entry.get("path", [])
        if not path_layers:
            print(f"Skipping case_id {case_id}: empty path.")
            skipped += 1
            continue

        l_attr = path_layers[-1]

        path_ok, path_inputs, l_attr_output = apply_path_and_check(
            model=model,
            clean_prompt=clean_prompt,
            counterfact_prompt=counterfact_prompt,
            expected_token=target_token,
            ent_slice=ent_slice,
            path_layers_indices=path_layers,
        )

        print(f"Case ID: {case_id}, Path OK: {path_ok}, Path Layers: {path_layers}, path_inputs keys: {list(path_inputs.keys())}, l_attr_output shape: {l_attr_output.shape if l_attr_output is not None else None}")

        counterfactual_inputs = collect_layer_inputs(model, counterfact_prompt, ent_slice)

        if not path_ok:
            print(f"Warning: path check failed for case_id {case_id}; running sufficiency test anyway.")

        is_correct = run_sufficiency_test(
            model=model,
            prompt=clean_prompt,
            expected_token=target_token,
            ent_slice=ent_slice,
            path_inputs=path_inputs,
            counterfactual_inputs=counterfactual_inputs,
            l_attr=l_attr,
            l_attr_rep=l_attr_output,
            intervention_type=args.exp_type,
        )

        total += 1
        if not is_correct:
            failing += 1

    failing_rate = (failing / total * 100.0) if total > 0 else 0.0

    print("\n--- Sufficiency experiment summary ---")
    print(f"Experiment type: {args.exp_type}")
    print(f"Path mode: {args.path_mode}")
    print(f"Total evaluated cases: {total}")
    print(f"Skipped cases: {skipped}")
    print(f"Failing cases: {failing}")
    print(f"Failing rate: {failing_rate:.2f}%")


if __name__ == "__main__":
    main()
