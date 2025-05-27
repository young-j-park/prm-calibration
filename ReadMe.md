# Think Smarter, Not Harder: Uncertainty Calibration of Process Reward Models

---

## Environment

- **Python** == 3.10  
- Huggingface
  - [vLLM](https://github.com/vllm-project/vllm)  
  - [transformers](https://github.com/huggingface/transformers)  
  - [datasets](https://github.com/huggingface/datasets)  
  - [trl](https://github.com/lvwerra/trl)  
  - [peft](https://github.com/huggingface/peft)  
- Other common packages: `torch`, `numpy`, `tqdm`, `GPUtil`, etc.

---

## Usage

Below are exemplary commands for running the scripts. 
**Note**: To handle large datasets, you can process them in multiple chunks (e.g., `0, 1, 2, 3, 4`). 
For brevity, we often show a single `chunk` argument. 
Feel free to loop over multiple chunks as needed. 
Also, LLM names (e.g., `meta-llama/Llama-3.2-1B-Instruct`) or PRM names `Qwen/Qwen2.5-Math-PRM-7B`) are placeholders and can be replaced with your desired (or fine-tuned) models.

## Calibration
### Step 1: Forward Inference (First Pass)

```
python main_infer_vllm.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --n_generations 8 \
    --use_cuda_graph \
    --hf_token {HF_TOKEN}
```

### Step 2: Monte Carlo Rollouts (Second Pass)

```
python main_rollout_vllm.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --dataset math500train \
    --chunk 0 \
    --n_generations 8 \
    --k_prefix_segments {0 1 2 3 ... 20} \
    --input_response_idx_begin 0 \
    --input_response_idx_end 8 \
    --use_cuda_graph \
    --hf_token {HF_TOKEN}
```

### Step 3:  Monte Carlo Estimate the Success Probabilities

```
python process_mc_accuracy.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --dataset math500train \
    --chunk 0 \
    --n_generations 8 \
    --max_prefix_steps 20 \
    --input_response_idx_begin 0 \
    --input_response_idx_end 8
```

### Step 4: Calibrate PRM

```
python finetune_mse_prm.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --model_name Qwen/Qwen2.5-Math-PRM-7B \
    --dataset math500train \
    --total_chunks 5 \
    --response_begin_idx 0 \
    --response_end_idx 8 \
    --loss_fn wql \
    --n_bins 9
```

## One-Pass Inference (e.g., for Best-of-N):
```
python main_infer_vllm.py \
    --mode onepass --n_generations 64 \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --use_cuda_graph \
    --hf_token {HF_TOKEN}
```

### Compute accuracy:
```
python evaluate_onepass_results.py \
    --n_generations 64 \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --prm_model_name Qwen/Qwen2.5-Math-PRM-7B \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --evaluate accuracy
```

### Compute PRM rewards:
```
python evaluate_onepass_results.py \
    --n_generations 64 \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --prm_model_name Qwen/Qwen2.5-Math-PRM-7B \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --evaluate reward
```

## Beam Search with Intance-Adaptive Sampling (IAS)
###  Beam Search with uncalibrated PRM:
```
python main_infer_beamsearch_vllm.py \
    --n_generations 64 --beam_width 8 --freq 5 \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --prm_model_name Qwen/Qwen2.5-Math-PRM-7B \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --use_cuda_graph
```

### Beam Search with calibrated PRM:
```
python main_infer_beamsearch_vllm.py \
    --n_generations 64 --beam_width 8 --freq 5 \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --prm_model_name Qwen/Qwen2.5-Math-PRM-7B \
    --prm_peft_dir {PEFT_DIR} \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --use_cuda_graph
```

### Beam Search with calibrated PRM + IAS-of-K:
```
python main_infer_beamsearch_vllm.py \
    --interleave_K --lb_idx 0 \
    --n_generations 64 --beam_width 8 --freq 5 \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --prm_model_name Qwen/Qwen2.5-Math-PRM-7B \
    --prm_peft_dir {PEFT_DIR} \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --use_cuda_graph
```

### Beam Search with calibrated PRM + IAS-of-M:
```
python main_infer_beamsearch_vllm.py \
    --interleave_M --lb_idx 0 \
    --n_generations 64 --beam_width 8 --freq 5 \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --prm_model_name Qwen/Qwen2.5-Math-PRM-7B \
    --prm_peft_dir {PEFT_DIR} \
    --dataset math500train \
    --chunk 0 --total_chunks 5 \
    --use_cuda_graph
```