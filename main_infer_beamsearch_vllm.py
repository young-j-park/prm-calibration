"""
Script to run Beam Search with Instance-Adaptive Scaling (IAS).
"""

import os
import argparse
import json

import torch
from peft import PeftModelForCausalLM

from prm import load_prm
from utils.data import get_dataset
from utils.llm import load_llm_with_retries, get_prompt_format
from utils.ias import determine_M, determine_K
from utils.beam_search import (
    get_scores,
    process_one_step,
    update_done,
    select_topk,
    aggregate,
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
        "--prm_model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        choices=[
            "peiyi9979/math-shepherd-mistral-7b-prm",
            "Qwen/Qwen2.5-Math-PRM-7B",
            "GAIR/ReasonEval-7B",
        ],
        help="Name or path of the reward model.",
    )
    parser.add_argument(
        "--prm_peft_dir",
        type=str,
        default=None,
        help="Path of the peft adapater for the reward model.",
    )
    parser.add_argument(
        "--lb_idx",
        type=int,
        default=0,
        help="0 - 10% / 1 - 50% / 2 - 90%",
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
        "--beam_width",
        type=int,
        default=8,
        help="Beam width."
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=5,
        help="Search Frequency."
    )
    parser.add_argument(
        "--interleave_K",
        action="store_true",
        help="Determine the number of particles adaptively using prm scores."
    )
    parser.add_argument(
        "--interleave_M",
        action="store_true",
        help="Determine the number of particles adaptively using prm scores."
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=64,
        help="Number of generation samples per prompt."
    )
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
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial ID"
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

    # 0. Check 2-GPU
    num_gpus = torch.cuda.device_count()
    assert num_gpus == 2
    gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"Detected {num_gpus} GPU(s): {gpu_ids}.")

    if num_gpus == 0:
        print("No NVIDIA GPUs found.")
        return

    # 1. Load model with retries
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])  # ensure the model use cuda:0
    llm = load_llm_with_retries(args)
    if "qwen" in args.model_id.lower():
        args.stop_ids = [151645, 151643]  # Some Qwen stop tokens
    else:
        args.stop_ids = None

    # 2. Load PRM
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    prm = load_prm(args.prm_model_name, device="cuda:1")  # assign to cuda:1
    _LB_IDX = None
    _MEDIAN_IDX = None
    if args.prm_peft_dir:
        _N_BINS = 9
        prm.model.resize_token_embeddings(len(prm.tokenizer) + _N_BINS)
        
        if "wql" in args.prm_peft_dir:
            _N_QUANTILES = 3
            _LB_IDX = args.lb_idx  # 10% or 50% or 100%
            _MEDIAN_IDX = 1  # 50%
            prm.convert_to_quantile_regression_head(_N_QUANTILES)
                        
        peft_model = PeftModelForCausalLM.from_pretrained(prm.model, args.prm_peft_dir)
        peft_model.eval()
    
    # 3. Load dataset
    dataset, q_key, a_key = get_dataset(
        dataset_name=args.dataset,
        chunk=args.chunk,
        total_chunks=args.total_chunks,
    )
    print(f"Dataset loaded. Number of samples: {len(dataset)}")

    # Determine which prompt format to use
    prompt_format = get_prompt_format(args.model_id)

    # Collect the questions and prompt
    formatted_prompts = []
    questions = []
    for i, sample in enumerate(dataset):
        user_prompt = prompt_format.replace("{input}", sample[q_key])
        formatted_prompts.append(user_prompt)
        questions.append(sample[q_key])

        if args.debug and len(questions) == 3:
            break
        
    # 4. Initialize Beam Search State
    _N_QUESTIONS = len(formatted_prompts)
    
    assert args.n_generations % args.beam_width == 0
    _K_MAX = args.n_generations // args.beam_width
    _M_MAX = args.beam_width
    
    done = [[False] for _ in range(_N_QUESTIONS)]               # N x answers
    stop_reasons = [[ "\n\n" ] for _ in range(_N_QUESTIONS)]    # N x answers
    output_reasonings = [[[]] for _ in range(_N_QUESTIONS)]     # N × 1 paths, each path is list[str]
    gen_counts = [[] for _ in range(_N_QUESTIONS)]              # N × steps
    gen_tok_counts = [[] for _ in range(_N_QUESTIONS)]          # N × steps
    
    tok = llm.get_tokenizer()

    # 5. Beam Search
    step = 0
    score_dict = {}
    while True:
        current_texts = [["\n\n".join(path) for path in paths] for paths in output_reasonings]  # N x answers

        if step == 0:
            n_samples = [[args.n_generations] for _ in range(_N_QUESTIONS)]
                
        elif step % args.freq == 0:
            # 1) score candidate reasonings
            scores = get_scores(prm, questions, current_texts, score_dict)  # N x answers
            
            # 2) pick among candidates: N x answers -> N x K
            if _LB_IDX is not None:
                lb_scores = [[s[_LB_IDX] for s in sc_row] for sc_row in scores]
                K = determine_K(lb_scores, _M_MAX, _K_MAX, interleave=args.interleave_K)
            else:
                K = determine_K(scores, _M_MAX, _K_MAX, interleave=args.interleave_K)
                    
            output_reasonings, current_texts, stop_reasons, scores, done = select_topk(
                output_reasonings, current_texts, stop_reasons, scores, done, K, _MEDIAN_IDX
            )
        
            # 3) decide how many to sample next step
            if _LB_IDX is not None:
                lb_scores = [[s[_LB_IDX] for s in sc_row] for sc_row in scores]
                n_samples = determine_M(lb_scores, _M_MAX, _K_MAX, interleave=args.interleave_M)
            else:
                n_samples = determine_M(scores, _M_MAX, _K_MAX, interleave=args.interleave_M)
                
        else:
            n_samples = [[1] * len(current_texts[i]) for i in range(_N_QUESTIONS)]
            
        # 4) prepare prefixes for next step
        formatted_inputs = [
            [formatted_prompts[i] + txt + ("\n\n" if step > 0 and stop == "\n\n" else "") for txt, stop in zip(row_texts, row_stops)]
            for i, (row_texts, row_stops) in enumerate(zip(current_texts, stop_reasons))
        ]
        
        # 5) update done flags
        done = update_done(done, stop_reasons)
        
        for i in range(_N_QUESTIONS):
            for j, text in enumerate(formatted_inputs[i]):
                total_tokens = len(tok.encode(text))
                if total_tokens >= args.max_model_len:
                    done[i][j] = True

        if all(all(r) for r in done) or step >= args.max_steps:
           break 

        # 6) one-step generation
        one_outs, one_stop_reasons = process_one_step(
            formatted_inputs, n_samples, done, llm, args
        )
        for i in range(_N_QUESTIONS):
            cnt = 0
            tok_cnt = 0
            for j in range(len(n_samples[i])):
                if not done[i][j]:
                    cnt += n_samples[i][j]
                    tok_cnt += sum([len(tok.encode(text)) for text in one_outs[i][j]])
            gen_counts[i].append(cnt)
            gen_tok_counts[i].append(tok_cnt)
            
        # 7) stitch into the reasoning paths
        output_reasonings, done, stop_reasons = aggregate(
            output_reasonings, done, one_outs, one_stop_reasons
        )
    
        step += 1

    print(f"Generated {_N_QUESTIONS} sets of generations.")

    # 6. Post-process (Pick the best again)
    all_results = [["\n\n".join(path) for path in paths] for paths in output_reasonings]
    all_scores = get_scores(prm, questions, all_results, score_dict)  # N x answers
    k_best = [1 for _ in range(_N_QUESTIONS)]
    _, final_results, _, _, _ = select_topk(
        output_reasonings, all_results, stop_reasons, all_scores, done, k_best, _MEDIAN_IDX
    )
    
    # 7. Save results
    # Each item in all_results is a list of length n_generations (strings).
    # Let's pair them up with the dataset question & answer for clarity.
    output_dir = os.path.join(
        args.output_dir,
        args.model_id.replace("/", "_"),  # avoid nested folders
        args.dataset
    )
    os.makedirs(output_dir, exist_ok=True)

    prm_nickname = args.prm_model_name.replace("/", "_")
    if args.prm_peft_dir:
        if "mse" in args.prm_peft_dir:
            prm_nickname += "_mse"
        else:
            prm_nickname += f"_wql{_LB_IDX}"

        prm_nickname += "_calibrated"

        if args.dataset not in args.prm_peft_dir:
            prm_nickname += "_transferred"

    tts_method = f"bs_{prm_nickname}_{args.n_generations}_{args.beam_width}_{args.freq}"

    if args.interleave_K:
        tts_method += "_interleave_K"
    if args.interleave_M:
        tts_method += "_interleave_M"
        
    out_fname = f"inference_{tts_method}_chunk_{args.chunk}_trial_{args.trial}.json"
    out_path = os.path.join(output_dir, out_fname)

    dataset_list = list(dataset)  # converting to standard Python list

    output_data = []
    for item, gens, cnts, tok_cnts in zip(dataset_list, final_results, gen_counts, gen_tok_counts):
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
