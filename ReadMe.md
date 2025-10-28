# Know What You Don’t Know: Uncertainty Calibration of Process Reward Models  


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.09338-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.09338)
[![🤗 HF Dataset](https://img.shields.io/badge/Dataset-prm_calibration-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/young-j-park/prm_calibration)
[![🤗 HF Collection](https://img.shields.io/badge/Models-calibrated--qwen2.5--math--prm--7b-blue?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/young-j-park/calibrated-qwen25-math-prm-7b-683fe3fa27858af86708e50d)


**Project Page:** [https://young-j-park.github.io/know-what-you-dont-know/](https://young-j-park.github.io/know-what-you-dont-know/)

</div>

<div align="center">
  <img src="https://young-j-park.github.io/know-what-you-dont-know/assets/images/motivation.png" alt="Instance-Adaptive Scaling" width="500"/>
  <p><em>Adaptive computation based on problem difficulty. Easy problems get quick decisions, hard problems get extended reasoning.</em></p>
</div>

## Overview

Process reward models (PRMs) guide inference-time scaling for LLMs, but even state-of-the-art PRMs can be **poorly calibrated** and **overestimate success probabilities**. We present:

- **📊 PRM Calibration via Quantile Regression**: Adjusts PRM outputs to align with true success probabilities
- **🧠 Instance-Adaptive Scaling (IAS)**: Dynamically adjusts inference budget based on estimated success likelihood
- **⚡ Up to 75% Computation Reduction**: While maintaining answer accuracy

### The Challenge: Overconfident PRMs

<div align="center">
  <img src="https://young-j-park.github.io/know-what-you-dont-know/assets/images/err_histogram_plot_uncalibrated_QwenPRM-7B_aimes.png" alt="PRM Calibration Error" width="350"/>
  <p><em>Uncalibrated PRMs systematically overestimate success probabilities, especially for weaker models and challenging problems.</em></p>
</div>

### Our Solution: Calibrated Confidence + Smart Allocation

<div align="center">
  <img src="https://young-j-park.github.io/know-what-you-dont-know/assets/images/prm.jpg" alt="PRM Calibration Process" width="500"/>
  <p><em>Our quantile regression calibration transforms existing PRMs to provide reliable uncertainty estimates with confidence intervals.</em></p>
</div>

---

## Installation

Create and activate a conda environment:
```bash
conda create -n prm-calibration python=3.10
conda activate prm-calibration
```

Install the required packages:
```bash
pip install torch==2.5.1
pip install vllm==0.7.2 transformers==4.48.2 trl==0.15.2 peft==0.14.0 GPUtil==1.4.0 latex2sympy2==1.9.1 word2number==1.1
```

### Package Versions

| Requirement   | Tested Version |
|---------------|-------------------|
| **Python**    | 3.10 |
| **PyTorch**   | 2.5.1 |
| **vLLM**      | 0.7.2 |
| **Hugging Face** | `transformers==4.48.2`, `trl==0.15.2`, `peft==0.14.0` |
| **Other Packages** | `GPUtil==1.4.0`, `latex2sympy2==1.9.1`, `word2number==1.1` |

## Quick Start

### Loading Calibration Dataset from HuggingFace
```python
from datasets import load_dataset

dataset = "math500"
policy_model_id_safe = "Llama-3.2-1B-Instruct"

# Load calibration dataset
ds = load_dataset(
    "young-j-park/prm_calibration", 
    data_files=f"{dataset}/{policy_model_id_safe}/data.json", 
    split="train"
)

# Access sample data
sample = ds[0]
question = sample["question"]
reasoning_prefix = sample["reasoning_prefix"]
success_prob = sample["success_prob"]
```

### Loading and Using PRM Models

#### Uncalibrated PRM
```python
from prm import load_prm

# Load PRM model
prm_model_id = "Qwen/Qwen2.5-Math-PRM-7B"
prm = load_prm(prm_model_id)

# Score a reasoning trajectory
uncalibrated_scores = prm.score([question], [[reasoning_prefix]])
prefix_reward = uncalibrated_scores[0][0][-1]  # last score
print(f"Uncalibrated PRM Reward: {prefix_reward}")
```

#### Calibrated PRM with Quantile Regression
```python
from peft import PeftModel

# Convert to quantile regression head
prm.convert_to_quantile_regression_head(M=3)  # outputs [0.1, 0.5, 0.9] quantiles

# Load calibrated weights
prm_model_id_safe = prm_model_id.split("/")[-1]
peft_model_id = f"young-j-park/{prm_model_id_safe}-calibrated-{policy_model_id_safe}"
peft_model = PeftModel.from_pretrained(prm.model, peft_model_id)

# Get calibrated quantile scores
calibrated_scores = prm.score([question], [[reasoning_prefix]])
prefix_reward_quantiles = calibrated_scores[0][0][-1]

print(f"10% Quantile: {prefix_reward_quantiles[0]}")
print(f"50% Quantile: {prefix_reward_quantiles[1]}")
print(f"90% Quantile: {prefix_reward_quantiles[2]}")
```

For a complete interactive demo, see our [HuggingFace Demo notebook](https://github.com/young-j-park/prm-calibration/blob/main/HuggingFace%20Demo.ipynb).

---

## Complete Usage

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

---

## Citation

```
@article{park2025calprm,
  title   = {Know What You Don't Know: Uncertainty Calibration of Process Reward Models},
  author  = {Park, Young-Jin and Greenewald, Kristjan and Alim, Kaveh and Wang, Hao and Azizan, Navid},
  journal = {arXiv preprint arXiv:2506.09338},
  year    = {2025}
}
```
