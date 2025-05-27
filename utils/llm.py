"""
Common LLM utility functions.
"""

import time
import gc

import torch
import GPUtil
from huggingface_hub import login
from vllm import LLM, SamplingParams

from utils.qwen_math_parser import contains_boxed


def login_hf(hf_token: str) -> None:
    """Login to Hugging Face Hub using a token."""
    login(token=hf_token)


def detect_gpus() -> (int, bool):
    """
    Detect and return:
        - The number of available GPUs.
        - Whether at least one of them is a V100.
    If no GPUs are found, the process is stopped.
    """
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s).")

    if num_gpus == 0:
        print("No NVIDIA GPUs found. Exiting.")
        exit(0)

    gpus = GPUtil.getGPUs()
    has_v100 = any("v100" in gpu.name.lower() for gpu in gpus)
    return num_gpus, has_v100


def init_llm(
    model_id: str,
    hf_token: str,
    seed: int = 42,
    gpu_memory_utilization: float = 0.95,
    use_cuda_graph: bool = False,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    dtype: str = "auto",
    tensor_parallel_size: int = None,
    max_model_len: int = 4096,
) -> LLM:
    """
    Initialize the vLLM model with given parameters.
    Automatically logs in to HF Hub.
    Automatically detects and configures for V100 if present.
    """
    # Hugging Face login
    login_hf(hf_token)

    # GPU detection
    num_gpus, has_v100 = detect_gpus()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus

    # If we find at least one V100, we adjust certain parameters
    if has_v100:
        dtype = "float16"  # forcibly use float16
        enable_prefix_caching = False
        enable_chunked_prefill = False

    llm = LLM(
        seed=seed,
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=(not use_cuda_graph),
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        dtype=dtype,
    )
    return llm


def load_llm_with_retries(args, max_trials: int = 5, delay: int = 1) -> LLM:
    """
    Attempt to initialize the LLM multiple times to handle
    intermittent errors (e.g., HF Hub issues, GPU OOM).
    """
    for attempt in range(max_trials):
        try:
            llm = init_llm(
                model_id=args.model_id,
                hf_token=args.hf_token,
                seed=args.seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
                use_cuda_graph=args.use_cuda_graph,
                enable_prefix_caching=args.enable_prefix_caching,
                enable_chunked_prefill=args.enable_chunked_prefill,
                dtype=args.dtype,
            )
            return llm
        except Exception as exc:
            print(f"LLM init attempt {attempt + 1} failed: {exc}")
            if attempt == max_trials - 1:
                raise RuntimeError("Failed to load LLM after maximum attempts.") from exc
            # Clean up memory between attempts
            if 'llm' in locals():
                del llm
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(delay)


def get_prompt_format(model_id: str):
    """
    Return the appropriate prompt format (system+user) for the given model_id.
    """
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
            "{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
            "<|im_start|>assistant\n"
        ),
    }

    if "qwen" in model_id.lower():
        return qeval["qwen-math-pf"]
    else:
        return qeval["llama-math-pf"]


def get_sampling_params(
        model_id: str,
        n_generations: int,
        temperature: float,
        max_tokens: int,
) -> SamplingParams:
    """
    Build a SamplingParams object with Qwen-specific or generic stop tokens,
    and generate ~25% more samples than needed so we can filter by 'boxed'.
    """
    if "qwen" in model_id.lower():
        stop_ids = [151645, 151643]  # Example stop tokens for Qwen
    else:
        stop_ids = None

    sampling_params = SamplingParams(
        n=n_generations,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_ids,
    )
    return sampling_params


def prioritize_boxed(outputs, n_generations: int):
    """
    Given vLLM generation outputs, reorder so that any result
    containing a boxed answer is first.
    Return only the top n_generations in that new order.
    """
    boxed = []
    non_boxed = []
    for out in outputs:
        if contains_boxed(out.text):
            boxed.append(out.text)
        else:
            non_boxed.append(out.text)
    return (boxed + non_boxed)[:n_generations]
