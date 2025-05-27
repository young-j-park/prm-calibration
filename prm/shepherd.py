
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn

from prm.reward import RewardModel


class MathShepherdPRM(RewardModel):

    """
    Another reward model (e.g. peiyi9979/math-shepherd-mistral-7b-prm).
    This uses + / - or step tag detection internally. 
    """
    def __init__(self, model_name="peiyi9979/math-shepherd-mistral-7b-prm", device="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
            
        
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'

        self.step_token = step_tag
        self.candidate_token_ids = self.tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
        self.step_token_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902
        self.quantile_regression = False

    def convert_to_quantile_regression_head(
        self,
        M: int,
    ):        
        old_fc = self.model.lm_head
        w_old = old_fc.weight.data
        in_dim = old_fc.in_features
        
        w_logistic = w_old[self.candidate_token_ids[0]] - w_old[self.candidate_token_ids[1]]
        
        new_fc = nn.Linear(in_dim, M, bias=True, dtype=old_fc.weight.dtype, device=old_fc.weight.device)
        
        with torch.no_grad():
            new_fc.weight.data[:] = w_logistic.unsqueeze(0).repeat(M, 1)
            
        self.model.lm_head = new_fc
        
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
                ans = ans.replace("\n\n", " ки\n\n")
                output = ans + " ки" if ans[-2:] != "\n\n" else ans

                input_for_prm = f"{question} {output}"
                input_id = torch.tensor([self.tokenizer.encode(input_for_prm)[:self.MAX_TOKENS]]).to(self.model.device)
            
                with torch.no_grad():
                    output = self.model(input_id).logits
                    
                    if self.quantile_regression:    
                        scores = output.sigmoid()
                    else:
                        logits = output[:,:,self.candidate_token_ids]
                        scores = logits.softmax(dim=-1)[:,:,0] 
                    step_scores = scores[input_id == self.step_token_id]
                    question_scores.append(step_scores.detach().cpu().tolist())

            all_scores.append(question_scores)
        return all_scores
