
import os
import argparse
import json
from typing import List, Tuple, Dict

import torch
from huggingface_hub import login
from vllm import LLM, SamplingParams
import GPUtil

from utils.data import get_dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inference script with vLLM.")

    # Model / HF Hub
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Model ID from Hugging Face Hub or local path."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face token for other models.",
        required=True,
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        choices=["math500", "math500train", "aime2024", "aime2025", "aime2025-2"],
        help="Which dataset to run on."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Processes only that chunk index from the dataset."
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=50,
        help="Total chunks to split the dataset into."
    )

    # Inference parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Max tokens to process."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Max steps to generate."
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=1,
        help="Number of generation samples per prompt."
    )

    # vLLM specific
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory used by vLLM."
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="Do not use CUDA graph."
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
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial ID."
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./infer_results",
        help="Directory where the inference results will be saved."
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode."
    )

    return parser.parse_args()


def _process_batch_one_step(
    batch_prompts: List[str],
    n_samples: int,
    llm: LLM,
    stop_ids: List[int],
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Run LLM.generate on a batch, return texts and their stop_reasons."""
    if not batch_prompts:
        return [], []
    params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        # stop=["\n\n", "<|eot_id|>"],
        stop=["<|eot_id|>"],
        stop_token_ids=stop_ids,
        seed=seed,
    )
    results = llm.generate(batch_prompts, params)
    outputs = [[out.text for out in res.outputs] for res in results]
    stop_reasons = [[out.stop_reason for out in res.outputs] for res in results]
    return outputs, stop_reasons


def process_one_step(
    formatted_inputs: List[List[str]],
    n_samples_grid: List[List[int]],
    done: List[List[bool]],
    llm: LLM,
    args: argparse.Namespace
) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
    """
    For each (question, path) not yet done, generate n_samples; 
    skip those already done.
    """
    n_q = len(formatted_inputs)
    # prep output placeholders
    all_outputs: List[List[List[str]]] = [[[] for _ in row] for row in formatted_inputs]
    all_reasons: List[List[List[str]]] = [[[] for _ in row] for row in formatted_inputs]

    # bucket by sample count
    to_batch: Dict[int, List[Tuple[int,int]]] = {}
    for i, (inp_row, ns_row, done_row) in enumerate(zip(formatted_inputs, n_samples_grid, done)):
        for j, (prompt, n_samp, is_done) in enumerate(zip(inp_row, ns_row, done_row)):
            if is_done:
                all_outputs[i][j] = [""]
                all_reasons[i][j] = [None]
            else:
                to_batch.setdefault(n_samp, []).append((i, j))

    # run each batch
    for samp_count, positions in to_batch.items():
        prompts = [formatted_inputs[i][j] for i,j in positions]
        outs, reasons = _process_batch_one_step(
            prompts, samp_count, llm,
            args.stop_ids, args.temperature, args.max_tokens, args.seed+args.trial
        )
        for (i, j), out_list, rsn_list in zip(positions, outs, reasons):
            all_outputs[i][j] = out_list
            all_reasons[i][j] = rsn_list

    return all_outputs, all_reasons


def aggregate(
    output_reasonings: List[List[List[str]]],
    step_outputs: List[List[List[str]]],
    step_reasons: List[List[List[str]]],
) -> Tuple[List[List[List[str]]], List[List[str]]]:
    """
    Take all existing paths and extend each by every new segment
    """
    agg_or, agg_sr = [], []
    for prev_paths, new_lists, new_reasons in zip(output_reasonings, step_outputs, step_reasons):
        next_paths, next_reas = [], []
        for path, candidates, reasons in zip(prev_paths, new_lists, new_reasons):
            for seg, rsn in zip(candidates, reasons):
                if """\\boxed""" in seg:
                    rsn = None
                
                if seg != "":
                    next_paths.append(path + [seg])
                else:
                    next_paths.append(path)
                    
                next_reas.append(rsn)
        agg_or.append(next_paths)
        agg_sr.append(next_reas)
    return agg_or, agg_sr
    

def main():
    """Main function for inference."""
    args = parse_args()

    # HF login for private or large models
    login(token=args.hf_token)

    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s).")

    if num_gpus == 0:
        print("No NVIDIA GPUs found.")
        return

    # Detect A100 vs V100
    gpus = GPUtil.getGPUs()
    v100 = False
    for gpu in gpus:
        if "v100" in gpu.name.lower():
            v100 = True
        
    # Initialize vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ensure the model use cuda:0
    llm = LLM(
        seed=args.seed + args.trial,
        model=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=(not args.use_cuda_graph),  # memory
        max_model_len=args.max_model_len,  # memory
        enable_prefix_caching=False if v100 else args.enable_prefix_caching, # v100
        enable_chunked_prefill=False if v100 else args.enable_chunked_prefill, # v100
        dtype="float16" if v100 else args.dtype # v100
    )

    # Prepare prompts
    qeval = {
        "llama-math-pf": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Solve the following math problem efficiently and clearly:\n\n"
            "- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n"
            "- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n"
            "## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n"
            "## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n"
            "Regardless of the approach, always conclude with:\n\n"
            "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
            "Where [answer] is just the final number or expression that solves the problem."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "{output}<|eot_id|>"
        ),
        "qwen-math-pf": (
            "<|im_start|>system\n"
            "Solve the following math problem efficiently and clearly:\n\n"
            "- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n"
            "- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n"
            "## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n"
            "## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n"
            "Regardless of the approach, always conclude with:\n\n"
            "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
            "Where [answer] is just the final number or expression that solves the problem."
            "<|im_end|>\n"
            "<|im_start|>user\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n",
            "{output}",
            "\n\n",
        ),
    }

    # Load dataset
    dataset, q_key, a_key = get_dataset(
        dataset_name=args.dataset,
        chunk=args.chunk,
        total_chunks=args.total_chunks,
    )
    print(f"Dataset loaded. Number of samples: {len(dataset)}")

    # Prepare sampling parameters
    if "qwen" in args.model_id.lower():
        args.stop_ids = [151645, 151643]  # Some Qwen stop tokens
    else:
        args.stop_ids = None

    # Determine which prompt format to use
    if "qwen" in args.model_id.lower():
        prompt_format = qeval["qwen-math-pf"]
    else:
        prompt_format = qeval["llama-math-pf"]

    # Collect the questions and prompt
    formatted_prompts = []
    questions = []
    for i, sample in enumerate(dataset):
        user_prompt = prompt_format[0].replace("{input}", sample[q_key])
        formatted_prompts.append(user_prompt)
        questions.append(sample[q_key])

        if args.debug and len(questions) == 3:
            break
        
    # INITIAL STATE
    N = len(formatted_prompts)
    done = [[False] for _ in range(N)]
    output_reasonings = [[[]] for _ in range(N)]     # N × 1 paths, each path is list[str]
    gen_counts = [[] for _ in range(N)]            # N × steps
    gen_tok_counts = [[] for _ in range(N)]              # N × steps
    
    tok = llm.get_tokenizer()
    
    n_samples = [[args.n_generations] for _ in range(N)]
    formatted_inputs = [ [prompt] for prompt in formatted_prompts]
    
    one_outs, one_reasons = process_one_step(
        formatted_inputs, n_samples, done, llm, args
    )
    
    for i in range(N):
        cnt = 0
        tok_cnt = 0
        for j in range(len(n_samples[i])):
            if not done[i][j]:
                cnt += 1
                tok_cnt += sum([len(tok.encode(text)) for text in one_outs[i][j]])
        gen_counts[i].append(cnt)
        gen_tok_counts[i].append(tok_cnt)
        
    # stitch into the reasoning paths
    output_reasonings, _ = aggregate(
        output_reasonings, one_outs, one_reasons
    )

    print(f"Generated {N} sets of generations.")

    # Post-process (Pick the best again)
    all_results = [["\n\n".join(path) for path in paths] for paths in output_reasonings]

    # Save results
    # Each item in all_results is a list of length n_generations (strings).
    # Let's pair them up with the dataset question & answer for clarity.
    output_dir = os.path.join(
        args.output_dir,
        args.model_id.replace("/", "_"),  # avoid nested folders
        args.dataset
    )
    os.makedirs(output_dir, exist_ok=True)

    out_fname = f"inference_one_pass_{args.n_generations}_chunk_{args.chunk}_trial_{args.trial}.json"
    out_path = os.path.join(output_dir, out_fname)

    dataset_list = list(dataset)  # converting to standard Python list

    output_data = []
    for item, gens, cnts, tok_cnts in zip(dataset_list, all_results, gen_counts, gen_tok_counts):
        output_data.append({
            "chunk": args.chunk,
            "total_chunks": args.total_chunks,
            "question": item[q_key],
            "gold_answer": item[a_key],
            "generations": gens,
            "generation_cnts": cnts,
            "generation_token_cnts": tok_cnts,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
