
import os
import json

import torch
import numpy as np
import pandas as pd

from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, concatenate_datasets


class DataCollatorForMaskedBinPositions(DataCollatorForLanguageModeling):
    """
    Custom data collator that extends DataCollatorForLanguageModeling to:
    compute the loss only on positions matching the bin tokens.
    """

    def __init__(self, tokenizer, step_token, bin_tokens, output_accuracy=False, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.step_token_id = tokenizer.encode(step_token)[-1]
        self.bin_token_ids = [tokenizer.encode(tok)[-1] for tok in bin_tokens]

    def torch_call(self, features):
        batch = super().torch_call(features)
        labels = batch["labels"]
        bin_token_tensor = torch.tensor(self.bin_token_ids, device=labels.device)
        mask = torch.isin(labels, bin_token_tensor)
        # Only positions matching bin tokens contribute to the loss.
        labels = labels.masked_fill(~mask, -100)
        batch["labels"] = labels
        return batch


def load_dataset(
        model_id: str,
        dataset_name: str,
        total_chunks: int,
        response_begin_idx: int,
        response_end_idx: int,
        bin_tokens: list,
        shuffle: bool = True,
        base_dir: str = "./infer_results",
        chunk: int = -1,
):
    """
    Load and concatenate dataset chunks from JSON files.

    Parameters:
      model_id: The model identifier (used to form directory names).
      dataset_name: The name of the dataset (or test dataset) to be used.
      total_chunks: Total number of chunks to load.
      response_begin_idx, response_end_idx: Indices to identify response boundaries.
      bin_tokens: List of tokens representing accuracy bins.
      shuffle: Whether to shuffle the final dataset.
      base_dir: Base directory for dataset files.

    Returns:
      A concatenated Hugging Face Dataset.
    """
    dataset_dir = os.path.join(base_dir, model_id.replace("/", "_"), dataset_name)
    bins = np.linspace(0.0, 1.0, len(bin_tokens))
    datasets = []

    chunk_list = range(total_chunks) if chunk == -1 else [chunk]
    for chunk_idx in chunk_list:
        file_name = (
            f"inference_chunk_{chunk_idx}_response_"
            f"{response_begin_idx}_{response_end_idx}_mc_accuracy.json"
        )
        dataset_file = os.path.join(dataset_dir, file_name)
        if not os.path.isfile(dataset_file):
            print(f"Warning: Dataset file {dataset_file} not found.")
            continue

        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build a DataFrame from the JSON data.
        df = pd.DataFrame(data, columns=["question", "first_pass_prefix", "k_prefix_segment",  "accuracy"])
        df = df[~df["first_pass_prefix"].isnull()].reset_index(drop=True)
        df["accuracy"] = df["accuracy"].astype(float)
        df["k_prefix_segment"] = df["k_prefix_segment"].astype(int)

        # Digitize the accuracy values and map to bin tokens.
        bin_ids = np.digitize(df["accuracy"].values, bins) - 1
        df["accuracy_token"] = [bin_tokens[int(bid)] for bid in bin_ids]

        datasets.append(Dataset.from_pandas(df))

    if not datasets:
        raise ValueError(f"No valid dataset chunks found in {dataset_dir}.")

    concat_ds = concatenate_datasets(datasets)
    return concat_ds.shuffle(seed=42) if shuffle else concat_ds


def get_formatting_func(model_name):
    """
    Return the appropriate prompt formatting function based on the model name.
    """
    mapping = {
        "peiyi9979/math-shepherd-mistral-7b-prm": _shepherd_formatting_prompts,
        "Qwen/Qwen2.5-Math-PRM-7B": _qwen_formatting_prompts,
        "GAIR/ReasonEval-7B": _reasoneval_formatting_prompts,
    }
    if model_name not in mapping:
        raise NotImplementedError(f"Model {model_name} is not implemented in get_formatting_func.")
    return mapping[model_name]


def _prepare_example(example):
    """Ensure the example is in dict-of-lists format."""
    if isinstance(example, dict):
        return {k: [v] for k, v in example.items()}
    return example


def _format_prompts(example, formatter):
    """
    General logic for formatting prompts. Applies the provided model-specific formatter.
    """
    example = _prepare_example(example)
    output_texts = []
    for q, prefix, acc_token in zip(
            example["question"], example["first_pass_prefix"], example["accuracy_token"]
    ):
        if prefix.endswith("\n\n"):
            prefix = prefix[:-2]
        output_texts.append(formatter(q, prefix, acc_token))
    return output_texts


def _shepherd_formatting_prompts(example, step_token="ки"):
    """Format prompts for the Shepherd model."""
    return _format_prompts(
        example,
        lambda q, prefix, acc_token: f"{q} {prefix} {step_token}{acc_token}"
    )


def _qwen_formatting_prompts(example, step_token="<extra_0>"):
    """Format prompts for the Qwen model."""
    system_prefix = (
        "<|im_start|>system\nPlease reason step by step, "
        "and put your final answer in \\boxed{}.<|im_end|>\n"
    )
    return _format_prompts(
        example,
        lambda q, prefix, acc_token: (
                system_prefix
                + f"<|im_start|>user\n{q}<|im_end|>\n"
                + f"<|im_start|>assistant\n{prefix}{step_token}{acc_token}<|im_end|><|endoftext|>"
        ),
    )


def _reasoneval_formatting_prompts(example, step_token='"<s>'):
    """Format prompts for a ReasonEval-based model."""
    return _format_prompts(
        example,
        lambda q, prefix, acc_token: (
            f"Question:\n{q}\nAnswer:\nLet's think step by step.\n"
            f"{prefix}{step_token}{acc_token}"
        ),
    )
