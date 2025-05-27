
import torch
import torch.nn as nn
from trl import SFTTrainer


class ShepherdBaseSFTTrainer(SFTTrainer):
    def __init__(self, candidate_token_ids, bin_token_ids, **training_args):
        """
        Base class that handles:
          - candidate_token_ids  (e.g. [good_id, bad_id])
          - bin_pivot_id & n_bins
          - check that bin_token_ids form a contiguous range
        Subclasses must implement `_calc_loss(scores, targets)`.
        """
        super().__init__(**training_args)
        self.candidate_token_ids = candidate_token_ids
        self.bin_pivot_id = bin_token_ids[0]
        self.n_bins = len(bin_token_ids) - 1

        # ensure bins are contiguous
        for i, bid in enumerate(bin_token_ids):
            assert bid == self.bin_pivot_id + i, (
                f"bin_token_ids must be contiguous, "
                f"but got pivot {self.bin_pivot_id} at idx {i} != {bid}"
            )

    def activate(self, logits):
        pass
        
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        mask = shift_labels != -100

        scores = self.activate(shift_logits[mask])
        targets = (shift_labels[mask] - self.bin_pivot_id).to(scores.dtype) / self.n_bins
        
        loss = self._calc_loss(scores, targets)
        
        return (loss, outputs) if return_outputs else loss


class ShepherdMSESFTTrainer(ShepherdBaseSFTTrainer):
    def __init__(self, quantiles, candidate_token_ids, bin_token_ids, **training_args):
        """
        quantiles: Dummy variable
        """
        super().__init__(candidate_token_ids, bin_token_ids, **training_args)
        self.loss_fn = nn.MSELoss()
        
    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[..., self.candidate_token_ids].softmax(dim=-1)[..., 0]
        
    def _calc_loss(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean‐Squared‐Error between predicted good‐token probability and normalized target."""
        return self.loss_fn(scores, targets)


class ShepherdQuantileSFTTrainer(ShepherdBaseSFTTrainer):
    def __init__(self, quantiles, candidate_token_ids, bin_token_ids, **training_args):
        """
        quantiles: 1D Tensor of quantiles, e.g. torch.tensor([0.1,0.5,0.9])
        """
        super().__init__(candidate_token_ids, bin_token_ids, **training_args)
        self.quantiles = quantiles
        self.candidate_token_ids = None
        
    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.sigmoid()

    def _calc_loss(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Implements the “pinball” quantile loss:
          1/N sum over samples and quantiles of max((q−1)*(t−s), q*(t−s))
        """
        # scores: (N,M); targets: (N,) → (N,1)
        t = targets.unsqueeze(1)        # (N, 1)
        s = scores                      # (N, M)
        u = t - s                       # (N, M)
        q = self.quantiles.view(1, -1)  # (1, M)
        pin = torch.max((q - 1) * u, q * u)
        return pin.mean()
