
from typing import List

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from prm.reward import RewardModel


class QwenPRM(RewardModel):
    """
    QWEN 2-Class Reward Model (e.g. Qwen/Qwen2.5-Math-PRM-7B)
    Expects 2-class logits: [neg, pos].
    """

    def __init__(self, model_name="Qwen/Qwen2.5-Math-PRM-7B", device="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        # We will look for the <extra_0> token to mark step boundaries
        self.step_token = "<extra_0>"
        self.candidate_token_ids = [1, 0]  # qwen outputs two logits for [bad, good].
        self.step_token_id = self.tokenizer.encode("<extra_0>")[0]
        self.quantile_regression = False

    def convert_to_quantile_regression_head(
        self,
        M: int,
    ):       
        head_model = self.model.score
        old_fc = head_model[-1]
        w_old, b_old = old_fc.weight.data, old_fc.bias.data
        in_dim = old_fc.in_features
        
        w_logistic = w_old[self.candidate_token_ids[0]] - w_old[self.candidate_token_ids[1]]
        b_logistic = b_old[self.candidate_token_ids[0]] - b_old[self.candidate_token_ids[1]]
        
        new_fc = nn.Linear(in_dim, M, dtype=old_fc.weight.dtype, device=old_fc.weight.device)
        
        with torch.no_grad():
            new_fc.weight.data[:] = w_logistic.unsqueeze(0).repeat(M, 1)
            new_fc.bias.data[:]   = b_logistic

        head_model[-1] = new_fc
        
        self.quantile_regression = True
    
    def score(
        self,
        questions: List[str],
        answers: List[List[str]],
    ) -> List[List[List[float]]]:

        """
        Returns step-by-step positive-probabilities for each answer.
        Each item in the returned triple-nested list is all step scores for one answer.
        """
        all_scores = []

        for question_idx, (question, answer_list) in tqdm(enumerate(zip(questions, answers))):
            question_scores = []

            for ans in answer_list:
                # You might have multi-step text separated by "\n\n" or some other delimiter
                steps_list = ans.split("\n\n")

                # Construct Qwen conversation
                messages = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer in \\boxed{}."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"}
                ]
                # Qwen’s utility for building chat prompts:
                conversation_str = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

                with torch.inference_mode():
                    input_ids = self.tokenizer.encode(
                        conversation_str,
                        return_tensors="pt"
                    )[..., :self.MAX_TOKENS].to(self.model.device)

                    outputs = self.model(input_ids=input_ids, use_cache=True)
                    # outputs[0] should be logits: (batch_size=1, seq_len, num_labels=2)

                    token_masks = (input_ids == self.step_token_id)  # shape: [1, seq_len], True at step boundaries
                    # Now compute the positive probability (index=1) at each step boundary
                    step_scores = self._make_step_rewards(outputs[0], token_masks)
                    # step_scores is List[List[float]] but since batch_size=1, we just get step_scores[0]
                    question_scores.append(step_scores[0])

            all_scores.append(question_scores)
        return all_scores

    def _make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """
        logits: shape [batch_size, seq_len, 2]
        token_masks: shape [batch_size, seq_len]
        Returns shape [batch_size, num_steps], each step is the probability of 'pos'.
        """
        if self.quantile_regression:            
            probabilities = logits.sigmoid()  # [batch_size, seq_len, N_QUANTILES]
        else:
            probabilities = F.softmax(logits, dim=-1)  # [batch_size, seq_len, 2]
        
        # Zero out positions that are not step separators
        probabilities = probabilities * token_masks.unsqueeze(-1)  # [bs, seq_len, 2]

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # shape: [seq_len, N_QUANTILES or 2]
            valid_positions = sample[sample.sum(dim=-1) != 0]  # steps only
            
            if not self.quantile_regression and valid_positions.size(-1) != 2:
                raise ValueError("Expected 2-class logits (neg, pos). Found something else.")

            if self.quantile_regression:
                positive_probs = valid_positions
            else:
                positive_probs = valid_positions[:, 1]  # the 'pos' index
            all_scores_res.append(positive_probs.cpu().tolist())
        return all_scores_res
