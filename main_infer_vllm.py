"""
Script to run LLM forward inference using vLLM (Calibration).
"""

import os
import argparse
import json

from utils.data import get_dataset
from utils.llm import (
    load_llm_with_retries,
    get_prompt_format,
    get_sampling_params,
    prioritize_boxed
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inference script with vLLM.")

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

    # Inference parameters
    parser.add_argument(
        "--mode",
        type=str,
        default="calibration",
        choices=["calibration", "onepass"],
        help="Which mode to run on."
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

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./infer_results",
        help="Directory to save inference results."
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode."
    )

    return parser.parse_args()


def main():
    """Main function for inference."""
    args = parse_args()

    # 1. Load model with retries
    llm = load_llm_with_retries(args)

    # 2. Load dataset
    dataset, q_key, a_key = get_dataset(
        dataset_name=args.dataset,
        chunk=args.chunk,
        total_chunks=args.total_chunks,
    )
    print(f"Loaded dataset: {args.dataset} | Number of samples: {len(dataset)}")

    # 3. Prepare sampling parameters
    if args.mode == "calibration":
        n_generations = int(args.n_generations * 1.25)
    else:
        n_generations = args.n_generations
    sampling_params = get_sampling_params(
        model_id=args.model_id,
        n_generations=n_generations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # 4. Select the prompt format
    prompt_format = get_prompt_format(args.model_id)

    # 5. Prepare the prompts
    #    prompt_format[0] = system+user template
    #    We'll inject each sample into {input} placeholder
    formatted_prompts = []
    for sample in dataset:
        user_prompt = prompt_format.replace("{input}", sample[q_key])
        formatted_prompts.append(user_prompt)

        if args.debug and len(formatted_prompts) == 3:
            break

    # 6. Inference
    results = llm.generate(formatted_prompts, sampling_params)

    # 7. Post-process the outputs. Prioritize "boxed" answers for the calibration set.
    all_results = []
    for res in results:
        if args.mode == "calibration":
            final_list = prioritize_boxed(res.outputs, args.n_generations)
        else:
            final_list = [out.text for out in res.outputs]
        all_results.append(final_list)

    print(f"Generated {len(all_results)} sets of generations.")

    # 8. Save results
    output_dir = os.path.join(
        args.output_dir,
        args.model_id.replace("/", "_"),  # avoid nested folders
        args.dataset
    )
    os.makedirs(output_dir, exist_ok=True)


    if args.mode == "calibration":
        out_fname = f"inference_chunk_{args.chunk}.json"
    else:
        out_fname = f"inference_onepass_chunk_{args.chunk}.json"
    out_path = os.path.join(output_dir, out_fname)

    dataset_list = list(dataset)
    output_data = []
    for item, gens in zip(dataset_list, all_results):
        output_data.append({
            "chunk": args.chunk,
            "total_chunks": args.total_chunks,
            "question": item[q_key],
            "gold_answer": item[a_key],
            "generations": gens,
            "generation_cnts": args.n_generations,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
