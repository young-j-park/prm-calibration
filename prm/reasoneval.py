
from typing import List

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import MistralModel, MistralPreTrainedModel, LlamaModel, LlamaPreTrainedModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from prm.reward import RewardModel


class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()  # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores

        
class ReasonEvalPRM(RewardModel):
    def __init__(self, model_name="GAIR/ReasonEval-7B", device="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ReasonEval_7B.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
        ).eval()
            
        self.step_token = f"{self.tokenizer.pad_token}"
        self.step_token_id = self.tokenizer.encode(f"{self.step_token}")[-1]
        self.candidate_token_ids = [0, 1, 2]  # [p_{negative}, p_{neutral}, p_{positive}]
        self.quantile_regression = False

    def convert_to_quantile_regression_head(
        self,
        M: int,
    ):       
        old_fc = self.model.score_head
        w0, w1, w2 = old_fc.weight.data
        w_init = ((w1 - w0) + (w2 - w0)) / 2.0
        
        new_fc = nn.Linear(old_fc.in_features, M, bias=True,
                           dtype=old_fc.weight.dtype,
                           device=old_fc.weight.device)
        with torch.no_grad():
            new_fc.weight.data[:] = w_init.unsqueeze(0).repeat(M, 1)

        self.model.score_head = new_fc
        
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

        PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"

        all_scores = []

        for question_idx, (question, answer_list) in tqdm(enumerate(zip(questions, answers))):
            question_scores = []

            for ans in answer_list:
                # You might have multi-step text separated by "\n\n" or some other delimiter
                combined_steps = ans.replace("\n\n", self.step_token)
                prompt = PROMPT_FORMAT.format(input=question)

                tokenized_result = self.tokenizer(
                    prompt + self.step_token + combined_steps + self.step_token
                )['input_ids'][:self.MAX_TOKENS]
                
                ## Separating labels and adjusting token IDs
                labeled_token_indices = []
                adjusted_token_ids = []
                separator_count = 0
                for idx, token_id in enumerate(tokenized_result):
                    if token_id == self.step_token_id:
                        labeled_token_indices.append(idx - 1 - separator_count)
                        separator_count += 1
                    else:
                        adjusted_token_ids.append(token_id)
                        
                adjusted_token_ids = [1] + adjusted_token_ids # Adjusting to recover the first token_ids of the sentences
                adjusted_token_ids=torch.tensor([adjusted_token_ids], device=self.model.device)
                labeled_token_indices = labeled_token_indices[2:]  # Adjusting to skip the first two separator (begining and endding of the questions)
                
                attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool, device=self.model.device)
                # Evaluating reasoning steps using ReasonEval
                with torch.no_grad():
                    reasoning_scores = self.model(adjusted_token_ids, attention_mask)[0, labeled_token_indices, :]
                    
                    ## score: [p_{negative}, p_{neutral}, p_{positive}]
                    ## S_{validity} = p_{neutral} + p_{positive}
                    ## S_{redundancy} = p_{neutral}
                    
                    if self.quantile_regression:    
                        step_scores = reasoning_scores.sigmoid()
                    else:
                        step_scores = torch.softmax(reasoning_scores, dim=-1)[:, 1:].sum(dim=-1)
                
                question_scores.append(step_scores.detach().cpu().tolist())
            
            all_scores.append(question_scores)
        return all_scores
        