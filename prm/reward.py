
from typing import List

import abc


class RewardModel(abc.ABC):
    
    """
    A base abstract class for all PRM/Reward Models.
    """
    
    MAX_TOKENS = 4096

    @abc.abstractmethod
    def score(
        self,
        questions: List[str],
        answers: List[List[str]],
    ) -> List[List[List[float]]]:
        """
        Returns:
            A triple-nested list of shape:
            [
               [   # for question 0
                   [step_score_0, step_score_1, ...],  # for answer 0
                   [step_score_0, step_score_1, ...],  # for answer 1
                   ...
               ],
               [   # for question 1
                   ...
               ],
               ...
            ]
        """
        pass
        
    @abc.abstractmethod
    def convert_to_quantile_regression_head(
        self,
        M: int,
    ):
        """
        Replace the final logit Linear head with an M-output Linear head,
        initialized so that sigmoid(new_head(x)) == original softmax_score(x) for each of the M outputs.
    
        Args:
            M:      Number of quantile outputs (heads).
        """
        pass