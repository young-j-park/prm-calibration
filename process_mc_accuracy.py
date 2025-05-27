"""
Script to process inference results (after second pass) and compute accuracy.
"""

import os
import json
import warnings
import argparse

from tqdm import tqdm
import numpy as np

from utils.qwen_math_parser import math_equal, extract_answer

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Process inference results and compute accuracy.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="math500train",
        choices=["math500", "math500train", "aime2024", "aime2025", "aime2025-2"],
        help="Which dataset to run on."
    )
    parser.add_argument(
        "--max_prefix_steps",
        type=int,
        default=20,
        help="Maximum number of prefix steps to process."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID from Hugging Face Hub or local path."
    )
    parser.add_argument(
        "--input_response_idx_begin",
        type=int,
        default=0,
        help="Beginning index for first-pass responses."
    )
    parser.add_argument(
        "--input_response_idx_end",
        type=int,
        default=8,
        help="Ending index (exclusive) for first-pass responses."
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=8,
        help="Number of generations used for second pass."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./infer_results",
        help="Base directory for inference results."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        required=True,
        help="Chunk index to process."
    )

    return parser.parse_args()


def process_results(
    dataset: str,
    max_prefix_steps: int,
    model_id: str,
    input_response_idx_begin: int,
    input_response_idx_end: int,
    n_generations: int,
    output_dir: str,
    chunk: int
):
    """
    Process inference results for a specified chunk and compute accuracy.
    Returns a list of dictionaries with the per-sample accuracy.
    """
    results = []
    model_id = model_id.replace("/", "_")

    # Loop over prefix steps
    for k_step in range(max_prefix_steps):
        rollout_json_fname = (
            f"inference_chunk_{chunk}_"
            f"response_{input_response_idx_begin}_{input_response_idx_end}_"
            f"rollout_after_{k_step}_steps.json"
        )
        rollout_file_path = os.path.join(
            output_dir,
            model_id,
            dataset,
            rollout_json_fname
        )

        if not os.path.exists(rollout_file_path):
            continue

        with open(rollout_file_path, "r", encoding="utf-8") as f:
            rollout_data = json.load(f)

        if len(rollout_data) % n_generations != 0:
            raise ValueError(
                f"Data length {len(rollout_data)} is not a multiple "
                f"of n_generations {n_generations}."
            )

        n_questions = len(rollout_data) // n_generations
        for question_id in tqdm(range(n_questions)):
            for i_gen in range(n_generations):
                rollout = rollout_data[question_id * n_generations + i_gen]
                n_reasoning_steps = len(
                    rollout["first_pass_generation"].split("\n\n")
                )

                # Compute ground truth accuracy
                answer = extract_answer(rollout["gold_answer"], "math")

                if n_reasoning_steps < k_step:
                    continue
                elif n_reasoning_steps == k_step:
                    acc = float(
                        math_equal(
                            extract_answer(
                                rollout["first_pass_generation"], "math"
                            ),
                            answer,
                        )
                    )
                else:
                    acc_list = []

                    # handle an old bug from the main_rollout_vlm.py
                    rollouts = rollout.get("second_pass_generations", [])
                    for r in rollouts:
                        response = extract_answer(r, "math")
                        is_correct = math_equal(response, answer)
                        acc_list.append(is_correct)
                    acc = np.mean(acc_list) if acc_list else 0.0

                results.append({
                    "answer": rollout["gold_answer"],
                    "question_id": question_id,
                    "question": rollout["question"],
                    "response_id": i_gen,
                    "first_pass_prefix": rollout["first_pass_prefix"],
                    "k_prefix_segment": rollout["k_prefix_segment"],
                    "accuracy": acc,
                })
    return results


def main():
    """Parse command-line arguments and process results."""
    args = parse_args()

    all_results = process_results(
        dataset=args.dataset,
        max_prefix_steps=args.max_prefix_steps,
        model_id=args.model_id,
        input_response_idx_begin=args.input_response_idx_begin,
        input_response_idx_end=args.input_response_idx_end,
        n_generations=args.n_generations,
        output_dir=args.output_dir,
        chunk=args.chunk,
    )

    # Save the results
    output_json_fname = (
        f"inference_chunk_{args.chunk}_response_{args.input_response_idx_begin}_"
        f"{args.input_response_idx_end}_mc_accuracy.json"
    )
    out_path = os.path.join(
        args.output_dir,
        args.model_id.replace("/", "_"),
        args.dataset,
        output_json_fname
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Accuracy results saved to {out_path}")


if __name__ == "__main__":
    main()
