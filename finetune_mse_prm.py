"""
Train or run inference with a LoRA-based PEFT model, using a custom data collator
that masks specified tokens. The code loads a pre-trained reward model (PRM),
constructs and trains/infers a LoRA PEFT model, and saves results to disk.
"""

import os
import uuid
import datetime
import json
import argparse

import GPUtil
import torch
from tqdm import tqdm
from peft import PeftModelForCausalLM, LoraConfig
from trl import SFTConfig

from prm import load_prm
from trainer.finetuning import (
    DataCollatorForMaskedBinPositions,
    get_formatting_func,
    load_dataset,
)
from trainer import (
    ShepherdMSESFTTrainer,
    ShepherdQuantileSFTTrainer,
    QwenMSESFTTrainer,
    QwenQuantileSFTTrainer,
    ReasonEvalMSESFTTrainer,
    ReasonEvalQuantileSFTTrainer
)


# Global constants
QUANTILES = [0.1, 0.5, 0.9]
N_QUANTILES = len(QUANTILES)


def create_experiment_id():
    """
    Generate a unique identifier and a timestamp for the experiment.
    Returns:
        timestamp (str): Current date/time formatted as YYYYmmdd_HHMMSS.
        experiment_uuid (str): 16-char hex from a generated UUID.
    """
    experiment_uuid = uuid.uuid4().hex[:16]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp, experiment_uuid


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train or infer a PEFT model using SFT with custom masking."
    )

    # Reward model / Tokenizer
    parser.add_argument(
        "--model_name",
        "--prm_model_name",
        dest="prm_model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        choices=[
            "Qwen/Qwen2.5-Math-PRM-7B",
            "peiyi9979/math-shepherd-mistral-7b-prm",
            "GAIR/ReasonEval-7B",
        ],
        help="Name or path of the reward model (PRM)."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model ID from Hugging Face Hub or local path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map for the model. Options: 'cpu', 'cuda', 'auto', etc."
    )

    # Finetuning Parameters
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="mse",
        choices=["mse", "wql"],
        help="Loss function to use: MSE or Weighted Quantile Loss (WQL)."
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=2,
        help="Rank parameter (r) for LoRA."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for LoRA adapters."
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="qv",
        choices=["qkv", "qv", "q"],
        help="Which attention projection modules to apply LoRA to."
    )
    parser.add_argument(
        "--no_lora_embed_tokens",
        action="store_true",
        help="Disable LoRA for the embedding layer if set."
    )
    parser.add_argument(
        "--lora_layers_to_transform",
        type=str,
        default="quarter",
        choices=["all", "half", "quarter", "last"],
        help="Select which transformer layers to apply LoRA to."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate."
    )

    # Dataset/chunk handling
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500train",
        choices=["math500train", "aime2024"],
        help="Which dataset to run on."
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="math500train",
        choices=["math500", "math500train", "aime2024", "aime2025", "aime2025-2"],
        help="Which dataset to run on."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=-1,
        help="Only process this chunk index from the dataset, or -1 for all."
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=5,
        help="Total number of dataset chunks."
    )
    parser.add_argument(
        "--response_begin_idx",
        type=int,
        default=0,
        help="Start index for responses in the dataset."
    )
    parser.add_argument(
        "--response_end_idx",
        type=int,
        default=8,
        help="End index (exclusive) for responses in the dataset."
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=9,
        help="Number of bins for digitizing accuracy values."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Run training or inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Checkpoint name for inference. Examples: 'epoch-0', 'last'."
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Experiment ID for saving/loading previous runs."
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode."
    )

    return parser.parse_args()


def train_main(args):
    """
    Train a LoRA-based PEFT model using the specified reward model and dataset.

    Args:
        args (argparse.Namespace): Configuration parameters.
    """
    # 1. Load the PRM model and tokenizer
    prm = load_prm(args.prm_model_name, args.device)
    step_token = prm.step_token
    tokenizer = prm.tokenizer

    # 2. Create bin tokens and add to the tokenizer
    bin_tokens = [f"<BIN_{i}>" for i in range(args.n_bins)]
    tokenizer.add_special_tokens({"additional_special_tokens": bin_tokens})
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Prepare the base model
    base_model = prm.model
    base_model.resize_token_embeddings(len(tokenizer))

    if args.loss_fn == "wql":
        prm.convert_to_quantile_regression_head(N_QUANTILES)
        print(f"Converted to output {N_QUANTILES} quantiles: {QUANTILES}")

    base_model.train()

    # 4. Define target modules for LoRA
    if args.prm_model_name == "peiyi9979/math-shepherd-mistral-7b-prm":
        target_modules = ["lm_head"]
    elif args.prm_model_name == "Qwen/Qwen2.5-Math-PRM-7B":
        target_modules = ["score.0", "score.2"]
    elif args.prm_model_name == "GAIR/ReasonEval-7B":
        target_modules = ["score_head"]
    else:
        raise NotImplementedError

    if args.lora_target_modules == "qkv":
        target_modules += ["q_proj", "k_proj", "v_proj"]
    elif args.lora_target_modules == "qv":
        target_modules += ["q_proj", "v_proj"]
    elif args.lora_target_modules == "q":
        target_modules += ["q_proj"]

    if not args.no_lora_embed_tokens:
        target_modules += ["embed_tokens"]

    # 5. Optionally restrict layers for LoRA
    if args.lora_layers_to_transform == "all":
        layers_to_transform = None
    elif args.lora_layers_to_transform in ["half", "quarter"]:
        jump = 2 if args.lora_layers_to_transform == "half" else 4
        if args.prm_model_name == "peiyi9979/math-shepherd-mistral-7b-prm":
            layers_to_transform = list(range(jump - 1, 32, jump))
        elif args.prm_model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            layers_to_transform = list(range(jump - 1, 28, jump))
        elif args.prm_model_name == "GAIR/ReasonEval-7B":
            layers_to_transform = list(range(jump - 1, 32, jump))
        else:
            raise NotImplementedError
    elif args.lora_layers_to_transform == "last":
        if args.prm_model_name == "peiyi9979/math-shepherd-mistral-7b-prm":
            layers_to_transform = [31]
        elif args.prm_model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            layers_to_transform = [27]
        elif args.prm_model_name == "GAIR/ReasonEval-7B":
            layers_to_transform = [31]
        else:
            raise NotImplementedError
    else:
        raise ValueError

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
    )
    peft_model = PeftModelForCausalLM(base_model, peft_config)

    # 6. Load dataset
    dataset = load_dataset(
        model_id=args.model_id,
        dataset_name=args.dataset,
        total_chunks=args.total_chunks,
        response_begin_idx=args.response_begin_idx,
        response_end_idx=args.response_end_idx,
        bin_tokens=bin_tokens,
        shuffle=True,
        chunk=-1,  # use all
    )

    # 7. Custom data collator
    collator = DataCollatorForMaskedBinPositions(tokenizer, step_token, bin_tokens)

    # 8. Detect if A100 GPU is present (use FP16)
    a100 = any("a100" in gpu.name.lower() for gpu in GPUtil.getGPUs())
    if a100:
        print("A100 Found! FP16 training will be used.")

    # 9. Create output directory
    timestamp, experiment_uuid = create_experiment_id()
    output_dir = os.path.join(
        "./peft",
        args.prm_model_name.replace("/", "_"),
        args.model_id.replace("/", "_"),
        args.dataset,
        args.loss_fn,
        f"experiment_{timestamp}_{experiment_uuid}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # 10. Save hyperparameter configuration
    args_path = os.path.join(output_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 11. Determine training hyperparameters
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 32
    max_seq_length = 2048

    # 12. Configure checkpoint saving strategy
    if args.debug or "aime" in args.dataset:
        save_kwargs = {"save_strategy": "epoch"}
    else:
        save_kwargs = {"save_strategy": "steps", "save_steps": 500}

    sft_config = SFTConfig(
        output_dir=output_dir,
        eval_strategy="no",
        num_train_epochs=3,
        learning_rate=args.lr,
        max_seq_length=max_seq_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=a100,
        **save_kwargs,
    )

    # 13. Instantiate the Trainer
    formatting_func = get_formatting_func(args.prm_model_name)
    bin_token_ids = collator.bin_token_ids

    if args.prm_model_name == "peiyi9979/math-shepherd-mistral-7b-prm" and args.loss_fn == "mse":
        sft_trainer_cls = ShepherdMSESFTTrainer
    elif args.prm_model_name == "peiyi9979/math-shepherd-mistral-7b-prm" and args.loss_fn == "wql":
        sft_trainer_cls = ShepherdQuantileSFTTrainer
    elif args.prm_model_name == "Qwen/Qwen2.5-Math-PRM-7B" and args.loss_fn == "mse":
        sft_trainer_cls = QwenMSESFTTrainer
    elif args.prm_model_name == "Qwen/Qwen2.5-Math-PRM-7B" and args.loss_fn == "wql":
        sft_trainer_cls = QwenQuantileSFTTrainer
    elif args.prm_model_name == "GAIR/ReasonEval-7B" and args.loss_fn == "mse":
        sft_trainer_cls = ReasonEvalMSESFTTrainer
    elif args.prm_model_name == "GAIR/ReasonEval-7B" and args.loss_fn == "wql":
        sft_trainer_cls = ReasonEvalQuantileSFTTrainer
    else:
        raise NotImplementedError

    trainer = sft_trainer_cls(
        torch.tensor(QUANTILES, dtype=prm.model.dtype, device=prm.model.device),
        prm.candidate_token_ids,
        bin_token_ids,
        model=peft_model,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # 14. Train
    trainer.train()


def inference_load_model(args):
    """
    Load the base PRM, tokenizer, and the LoRA PEFT model from the specified checkpoint.

    Args:
        args (argparse.Namespace): Configuration parameters.

    Returns:
        tuple: (prm, peft_model, peft_dir, tokenizer, bin_tokens)
    """
    prm = load_prm(args.prm_model_name, args.device)
    tokenizer = prm.tokenizer

    # Create bin tokens and add them to the tokenizer
    bin_tokens = [f"<BIN_{i}>" for i in range(args.n_bins)]
    tokenizer.add_tokens(bin_tokens, special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare the base model
    base_model = prm.model
    base_model.resize_token_embeddings(len(tokenizer))

    if args.loss_fn == "wql":
        prm.convert_to_quantile_regression_head(N_QUANTILES)
        print(f"Converted to output {N_QUANTILES} quantiles: {QUANTILES}")

    # Determine the PEFT directory
    peft_dir = os.path.join(
        "./peft",
        args.prm_model_name.replace("/", "_"),
        args.model_id.replace("/", "_"),
        args.dataset,
        args.loss_fn,
    )
    if args.experiment_id is not None:
        peft_dir = os.path.join(peft_dir, f"experiment_{args.experiment_id}")

    # Resolve checkpoint directory
    if args.checkpoint.startswith("epoch-") or args.checkpoint in ["first", "last"]:
        prefix = "checkpoint-"
        checkpoints = sorted(
            int(path[len(prefix):])
            for path in os.listdir(peft_dir)
            if path.startswith(prefix)
        )
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {peft_dir}.")
        if args.checkpoint == "first":
            checkpoint = checkpoints[0]
        elif args.checkpoint == "last":
            checkpoint = checkpoints[-1]
        else:
            # e.g. args.checkpoint="epoch-5"
            chk_str = args.checkpoint[len("epoch-"):]
            checkpoint = int(chk_str)
            if checkpoint not in checkpoints:
                raise ValueError(f"Checkpoint {checkpoint} not found in {peft_dir}.")
        peft_dir = os.path.join(peft_dir, f"checkpoint-{checkpoint}")
    else:
        # e.g. a literal checkpoint name
        peft_dir = os.path.join(peft_dir, f"checkpoint-{args.checkpoint}")

    print("Loaded from:", peft_dir)

    peft_model = PeftModelForCausalLM.from_pretrained(base_model, peft_dir)
    peft_model.eval()
    return prm, peft_model, peft_dir, tokenizer, bin_tokens


def inference_main(args, prm, peft_model, peft_dir, tokenizer, bin_tokens):
    """
    Run inference with a PEFT model on the specified dataset and save the results.

    Args:
        args (argparse.Namespace): Configuration parameters.
        prm: Loaded reward model (with step token).
        peft_model: The LoRA-based PEFT model for inference.
        peft_dir (str): Path to the PEFT checkpoint directory.
        tokenizer: Model tokenizer.
        bin_tokens (list): List of bin token strings.
    """
    step_token = prm.step_token
    step_token_id = prm.step_token_id

    # 1. Load dataset
    dataset = load_dataset(
        model_id=args.model_id,
        dataset_name=args.test_dataset,
        chunk=args.chunk,
        total_chunks=args.total_chunks,
        response_begin_idx=args.response_begin_idx,
        response_end_idx=args.response_end_idx,
        bin_tokens=bin_tokens,
        shuffle=False,
    )

    # 2. Format prompts
    formatting_func = get_formatting_func(args.prm_model_name)
    formatted_prompts = formatting_func(dataset, step_token=step_token)

    # 3. Precompute bin token IDs
    bin_token_ids = [tokenizer.encode(tok)[-1] for tok in bin_tokens]

    results = []
    for i in tqdm(range(len(dataset))):
        acc = dataset["accuracy"][i]
        k_prefix_segment = dataset["k_prefix_segment"][i]
        prompt = formatted_prompts[i]

        inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
        input_ids = inputs["input_ids"][0]
        step_positions = (input_ids == step_token_id).nonzero(as_tuple=True)[0]

        if len(step_positions) == 0:
            print("Warning: Step token not found in the prompt.")
            results.append({
                "accuracy": float(acc),
                "prompt": prompt,
                "bin_probs": None,
                "prm_score": None,
                "k_prefix_segment": k_prefix_segment,
                "checkpoint": peft_dir,
            })
            continue

        # Use the last occurrence of the step token
        step_idx = step_positions[-1].item()

        # Special handling for GAIR/ReasonEval
        if args.prm_model_name == "GAIR/ReasonEval-7B":
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"][0:1, :step_idx], inputs["input_ids"][0:1, step_idx+1:]),
                dim=1
            )
            inputs["attention_mask"] = torch.cat(
                (
                    inputs["attention_mask"][0:1, :step_idx],
                    inputs["attention_mask"][0:1, step_idx+1:]
                ),
                dim=1
            )
            step_idx -= 1

        # Weighted Quantile Loss inference
        if args.loss_fn == "wql":
            with torch.no_grad():
                outputs = peft_model(**inputs)
                if args.prm_model_name == "GAIR/ReasonEval-7B":
                    output_logits = outputs
                else:
                    output_logits = outputs.logits

                pred_probs = output_logits[0, step_idx].sigmoid()

            results.append({
                "accuracy": float(acc),
                "prompt": prompt,
                "checkpoint": peft_dir,
                **{f"pred_prob[{q}]": p.item() for q, p in zip(QUANTILES, pred_probs)}
            })

        # MSE / classification-based inference
        else:
            with torch.no_grad():
                outputs = peft_model(**inputs)

                if args.prm_model_name == "GAIR/ReasonEval-7B":
                    pred_logits = outputs[0, step_idx]
                    pred_prob = torch.softmax(pred_logits, dim=0)[1:].sum()
                else:
                    pred_logits = outputs.logits[0, step_idx, prm.candidate_token_ids]
                    pred_prob = torch.softmax(pred_logits, dim=-1)[0]

            # Evaluate the base PRM scores with adapters disabled
            with torch.no_grad(), peft_model.disable_adapter():
                outputs_prm = peft_model(**inputs)
                if args.prm_model_name == "GAIR/ReasonEval-7B":
                    logits_prm = outputs[0, step_idx]
                    prm_score = torch.softmax(logits_prm, dim=0)[1:].sum()
                else:
                    logits_prm = outputs_prm.logits[0, step_idx, prm.candidate_token_ids]
                    prm_score = torch.softmax(logits_prm, dim=-1)[0]

            results.append({
                "accuracy": float(acc),
                "prompt": prompt,
                "pred_logits": pred_logits.detach().cpu().tolist(),
                "pred_prob": pred_prob.item(),
                "prm_logits": logits_prm.detach().cpu().tolist(),
                "prm_score": prm_score.item(),
                "k_prefix_segment": k_prefix_segment,
                "checkpoint": peft_dir,
            })

    # 4. Save inference results
    out_dir = os.path.join(
        "./infer_results", args.model_id.replace("/", "_"), args.test_dataset
    )
    os.makedirs(out_dir, exist_ok=True)

    chunk_label = args.chunk if args.chunk != -1 else "all"

    if args.experiment_id is not None:
        file_name = (
            f"inference_chunk_{chunk_label}_response_{args.response_begin_idx}_"
            f"{args.response_end_idx}_ft_{args.dataset}_"
            f"{args.experiment_id}_{args.checkpoint}_bin_probs.json"
        )
    else:
        file_name = (
            f"inference_chunk_{chunk_label}_response_{args.response_begin_idx}_"
            f"{args.response_end_idx}_ft_{args.dataset}_{args.checkpoint}_bin_probs.json"
        )

    out_path = os.path.join(out_dir, file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference results saved to {out_path}")


def main():
    """
    Main entry point:
      - Parse arguments
      - Run training or inference based on args.mode
    """
    args = parse_args()

    if args.mode == "train":
        train_main(args)
    else:
        prm, peft_model, peft_dir, tokenizer, bin_tokens = inference_load_model(args)
        inference_main(args, prm, peft_model, peft_dir, tokenizer, bin_tokens)


if __name__ == "__main__":
    main()
