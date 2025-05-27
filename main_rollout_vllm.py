"""
Script to run Monte Carlo rollouts (second pass).
"""

import os
import argparse
import json

from utils.qwen_math_parser import contains_boxed
from utils.llm import (
    load_llm_with_retries,
    get_prompt_format,
    get_sampling_params,
    prioritize_boxed
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Monte Carlo rollout script with vLLM.")

    # Model / HF Hub
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID from Hugging Face Hub or local path."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face token for private models."
    )

    # Dataset
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
        default=0,
        help="Process only this chunk index from the dataset."
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=5,
        help="Total chunks to split the dataset into."
    )

    # Rollout parameters
    parser.add_argument(
        "--k_prefix_segments",
        type=int,
        nargs="+",
        required=True,
        help="Number of segments to keep from first-pass generation (split by '\\n\\n')."
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=8,
        help="Number of generation samples per prompt."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate."
    )

    # vLLM parameters
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory used by vLLM."
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="Use CUDA graph if possible."
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching for large LLMs in vLLM."
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        help="Enable chunked prefill for large LLMs in vLLM."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type (e.g., auto, float16, bf16, float32)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    # Input / output
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./infer_results",
        help="Directory of the first-pass inference results."
    )
    parser.add_argument(
        "--input_response_idx_begin",
        type=int,
        default=0,
        help="Start index of first-pass generation for each question."
    )
    parser.add_argument(
        "--input_response_idx_end",
        type=int,
        default=8,
        help="End index of first-pass generation for each question."
    )

    return parser.parse_args()


def read_first_pass_data(args):
    """Read the JSON file produced by the first-pass inference."""
    input_dir = os.path.join(
        args.input_dir,
        args.model_id.replace("/", "_"),
        args.dataset
    )
    input_json_fname = f"inference_chunk_{args.chunk}.json"
    input_json_path = os.path.join(input_dir, input_json_fname)

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from first-pass inference: {input_json_path}")
    return data, input_dir


def second_pass_inference(args, llm, first_pass_data, k_prefix_segment, output_dir):
    """
    Perform the second-pass inference using the first-pass generation's prefix.
    The prefix is determined by k_prefix_segment segments split by "\n\n".
    """
    sampling_params = get_sampling_params(
        model_id=args.model_id,
        n_generations=int(args.n_generations * 1.25),
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    prompt_format = get_prompt_format(args.model_id)

    # Output file naming
    i_begin = args.input_response_idx_begin
    i_end = args.input_response_idx_end
    output_json_fname = (
        f"inference_chunk_{args.chunk}_response_{i_begin}_{i_end}_rollout_after_{k_prefix_segment}_steps.json"
    )
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_json_path = os.path.join(tmp_dir, output_json_fname)
    final_json_path = os.path.join(output_dir, output_json_fname)

    # If final file already exists, skip
    if os.path.exists(final_json_path):
        print(f"File already exists, skipping: {final_json_path}")
        return

    # Otherwise, we either resume from the tmp JSON or start fresh
    second_pass_results = []
    questions_done_count = {}

    if os.path.exists(tmp_json_path):
        # Resume from partial results
        with open(tmp_json_path, "r", encoding="utf-8") as f:
            second_pass_results = json.load(f)
        # Track how many times each question appears
        for item in second_pass_results:
            q = item["question"]
            questions_done_count[q] = questions_done_count.get(q, 0) + 1

    n_first_pass_response = args.input_response_idx_end - args.input_response_idx_begin

    # Start the second-pass inference
    for idx, item in enumerate(first_pass_data):
        question = item["question"]
        gold_answer = item["gold_answer"]
        # If the question already has all its possible expansions, skip
        if questions_done_count.get(question, 0) == n_first_pass_response:
            print(f"Skip question index={idx} (already processed).")
            continue

        # Extract relevant first-pass completions
        first_pass_generations = item["generations"][i_begin:i_end]

        # Build the user prompt from the chosen prompt format
        user_prompt_template = prompt_format.replace("{input}", question)

        # Iterate over each of the selected first-pass completions
        for gen_text in first_pass_generations:
            # Split by "\n\n" to determine prefix
            splitted = gen_text.split("\n\n")
            # If not enough segments, store an empty second-pass
            if len(splitted) <= k_prefix_segment:
                second_pass_results.append({
                    "question": question,
                    "gold_answer": gold_answer,
                    "first_pass_generation": gen_text,
                    "first_pass_prefix": None,
                    "k_prefix_segment": k_prefix_segment,
                    "second_pass_generations": [],
                })
                continue

            prefix_segments = splitted[:k_prefix_segment]
            prefix_text = "\n\n".join(prefix_segments) + "\n\n"
            new_prompt = user_prompt_template + prefix_text
            # If already boxed, store an empty second-pass
            if contains_boxed(prefix_text):
                second_pass_results.append({
                    "question": question,
                    "gold_answer": gold_answer,
                    "first_pass_generation": gen_text,
                    "first_pass_prefix": None,
                    "k_prefix_segment": k_prefix_segment,
                    "second_pass_generations": [],
                })
                continue

            # Generate second pass
            results = llm.generate([new_prompt], sampling_params)
            # Re-prioritize by "boxed"
            final_outputs = prioritize_boxed(results[0].outputs, args.n_generations)

            second_pass_results.append({
                "question": question,
                "gold_answer": gold_answer,
                "first_pass_generation": gen_text,
                "first_pass_prefix": prefix_text,
                "k_prefix_segment": k_prefix_segment,
                "second_pass_generations": final_outputs,
            })

        # Periodically save partial results
        if idx > 0 and idx % 10 == 0:
            with open(tmp_json_path, "w", encoding="utf-8") as f:
                json.dump(second_pass_results, f, ensure_ascii=False, indent=2)

    # Save the final results
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(second_pass_results, f, ensure_ascii=False, indent=2)

    print(f"Second-pass results saved to '{final_json_path}'")


def main():
    """Main entry point for the script."""
    args = parse_args()

    # 1. Load the model with retries
    llm = load_llm_with_retries(args)

    # 2. Read first-pass data
    first_pass_data, input_dir = read_first_pass_data(args)

    # 3. Run second-pass (rollout) for each prefix step
    for k_segment in args.k_prefix_segments:
        second_pass_inference(
            args,
            llm,
            first_pass_data,
            k_prefix_segment=k_segment,
            output_dir=input_dir  # save on the same directory; you can fix this.
        )


if __name__ == "__main__":
    main()
