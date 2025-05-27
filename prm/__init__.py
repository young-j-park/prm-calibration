
from prm.reward import RewardModel
from prm.qwen import QwenPRM
from prm.shepherd import MathShepherdPRM
from prm.reasoneval import ReasonEvalPRM


def load_prm(model_name: str, device="auto") -> RewardModel:
    """
    Returns an instance of the correct RewardModel class based on the model_name.
    """
    # You can do more sophisticated checks, if needed.
    if model_name == "Qwen/Qwen2.5-Math-PRM-7B":
        return QwenPRM(model_name, device=device)
    elif model_name == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherdPRM(model_name, device=device)
    elif model_name == "GAIR/ReasonEval-7B":
        return ReasonEvalPRM(model_name, device=device)
    else:
        raise ValueError(f"Unrecognized or unsupported reward model: {model_name}")
