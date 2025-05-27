"""
Script to compute math inference accuracies across multiple chunks and trials.
"""

import os
import json
import warnings
import argparse

import numpy as np
from tqdm import tqdm
from peft import PeftModelForCausalLM

from prm import load_prm
from utils.qwen_math_parser import math_equal, extract_answer


def parse_args():
    """
    Parse command-line arguments for computing inference accuracies or rewards
    on math datasets.
    """
    parser = argparse.ArgumentParser(
        description="Compute inference accuracies/rewards for math dataset."
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default="accuracy",
        choices=["accuracy", "reward"],
        help="Which metric to compute: accuracy or reward."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="math500train",
        choices=["math500", "math500train", "aime2024", "aime2025", "aime2025-2"],
        help="Which dataset to run on."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=-1,
        help="Number of chunks to process. Set -1 for all chunks."
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=5,
        help="Total number of chunks to process."
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=64,
        help="Number of generations per example."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID from Hugging Face Hub or local path."
    )
    parser.add_argument(
        "--prm_model_name",
        type=str,
        default=None,
        help="PRM model identifier."
    )
    parser.add_argument(
        "--prm_peft_dir",
        type=str,
        default=None,
        help="Path of the PEFT adapter for the reward model."
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="./infer_results",
        help="Directory containing inference result subfolders."
    )
    return parser.parse_args()


def load_inference_data(args):
    """
    Load inference JSON data across the specified chunks for the given dataset.
    Returns a dictionary of {dataset_name: list_of_inference_data}.
    """
    total_chunks = args.total_chunks
    chunk_list = list(range(total_chunks)) if args.chunk == -1 else [args.chunk]

    # Replace '/' with '_' in model_id to avoid filesystem issues.
    model_id_sanitized = args.model_id.replace("/", "_")

    # Decide which dataset(s) to process
    if args.dataset == "aimes":
        dataset_list = ["aime2024", "aime2025", "aime2025-2"]
    elif args.dataset == "aime2025s":
        dataset_list = ["aime2025", "aime2025-2"]
    else:
        dataset_list = [args.dataset]

    # Collect data for each dataset
    infer_data_dict = {}
    for dataset_name in dataset_list:
        combined_data = []
        for chunk_idx in chunk_list:
            json_path = os.path.join(
                args.results_dir,
                model_id_sanitized,
                dataset_name,
                f"inference_onepass_chunk_{chunk_idx}.json"
            )
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            combined_data.extend(data)
        infer_data_dict[dataset_name] = combined_data

    return infer_data_dict


def compute_accuracies(args, infer_data_dict):
    """
    Compute accuracy for each dataset and save results to a .npy file.
    Accuracy is computed by checking if the predicted answer equals the gold answer.
    """
    M = args.n_generations
    model_id_sanitized = args.model_id.replace("/", "_")
    chunk_str = "all" if args.chunk == -1 else f"chunk_{args.chunk}"

    for dataset_name, infer_data in infer_data_dict.items():
        accs = []
        for dat in tqdm(infer_data):
            gold_answer = extract_answer(dat["gold_answer"], "math")
            for i in range(M):
                prediction = extract_answer(dat["generations"][i], "math")
                acc = float(math_equal(prediction, gold_answer))
                accs.append(acc)
        accs = np.array(accs).reshape((-1, M))

        # Save accuracy array to .npy
        output_path = os.path.join(
            args.results_dir,
            model_id_sanitized,
            dataset_name,
            f"inference_onepass_{chunk_str}_accuracy.npy"
        )
        np.save(output_path, accs)


def compute_rewards(args, infer_data_dict):
    """
    Compute reward scores for each dataset and save results to a .npy file.
    Rewards are computed via a loaded PRM (possibly with PEFT adapter).
    """
    model_id_sanitized = args.model_id.replace("/", "_")
    chunk_str = "all" if args.chunk == -1 else f"chunk_{args.chunk}"
    if args.prm_model_name is None:
        raise ValueError("Must specify a PRM model name.")
    prm_model_name_sanitized = args.prm_model_name.replace("/", "_")


    # We'll load the PRM model once (the first dataset) and reuse
    prm = None
    peft_model = None

    # For each dataset, compute and save rewards
    dataset_list = list(infer_data_dict.keys())
    for i, dataset_name in enumerate(dataset_list):

        # Adjust PRM nickname according to peft_dir details
        prm_nickname = prm_model_name_sanitized
        if args.prm_peft_dir:
            if "mse" in args.prm_peft_dir:
                prm_nickname += "_mse"
            prm_nickname += "_calibrated"

            if dataset_name not in args.prm_peft_dir:
                prm_nickname += "_transferred"  # ood

        output_path = os.path.join(
            args.results_dir,
            model_id_sanitized,
            dataset_name,
            f"inference_onepass_{chunk_str}_{prm_nickname}.npy"
        )

        # Skip if rewards file already exists
        if os.path.exists(output_path):
            continue

        # Load PRM model if not loaded yet
        if i == 0:
            prm = load_prm(args.prm_model_name)
            if args.prm_peft_dir:
                _N_BINS = 9
                prm.model.resize_token_embeddings(len(prm.tokenizer) + _N_BINS)

                if "wql" in args.prm_peft_dir:
                    _N_QUANTILES = 3
                    prm.convert_to_quantile_regression_head(_N_QUANTILES)

                peft_model = PeftModelForCausalLM.from_pretrained(
                    prm.model, args.prm_peft_dir
                )
                peft_model.eval()

        # Prepare data for reward scoring
        infer_data = infer_data_dict[dataset_name]
        questions = []
        predictions = []
        infer_data = infer_data_dict[dataset_name]
        for dat in infer_data:
            questions.append(dat["question"])

            # Estimate the reward / success probability for the generations + question
            predictions.append(dat["generations"] + [""])
        rewards = prm.score(questions, predictions)
        rewards = np.array([[rs[-1] for rs in row] for row in rewards])

        # save
        np.save(output_path, rewards)


def main():
    """
    Main entry point. Loads data, then computes either accuracy or reward scores.
    """
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. Load inference data across chunks
    infer_data_dict = load_inference_data(args)

    # 2. Evaluate based on selected metric
    if args.evaluate == "accuracy":
        compute_accuracies(args, infer_data_dict)
    elif args.evaluate == "reward":
        compute_rewards(args, infer_data_dict)


if __name__ == "__main__":
    main()
