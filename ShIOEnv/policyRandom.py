import torch
from torch import nn
from typing import Tuple, List, Dict, Optional


class PolicyRandom(nn.Module):
    """
    A "random" policy that ignores the sequence entirely and
    produces a uniform distribution (logits=0.0) over valid actions,
    with invalid actions masked out to a large negative value.
    Shares the same interface and mask‐buffer setup as PolicyNetwork.
    """
    def __init__(
        self,
        vocab_size: int,
        config: dict,
        production_masks: Dict[str, torch.Tensor],
        cmd_key: str,
        max_action_dim: int
    ):
        super().__init__()
        self.cmd_actor_head_key = cmd_key

        # Map each head key to a row index, and keep a reverse list
        self.head2idx = {k: i for i, k in enumerate(production_masks.keys())}

        self.MAX_ACTION_DIM = max_action_dim
        self.MASK_VALUE = -1e5

        # Build one giant mask tensor: 0 keeps a logit, -1e5 blanks it out when softmaxed by torch.Categorical
        all_masks = torch.full(
            (len(production_masks), self.MAX_ACTION_DIM),
            fill_value=self.MASK_VALUE,
            dtype=torch.float32,
        )

        for head_key, mask in production_masks.items():
            row = self.head2idx[head_key]
            keep = mask.to(torch.bool)
            all_masks[row, keep] = 0.0  # zero means “keep logit”
        # Put on the same device as the model but mark non‑persistent
        self.register_buffer("all_action_masks", all_masks, persistent=False)

    @torch.no_grad()
    def _validate_inputs(self, x: torch.Tensor):
        if torch.any(x < 0):
            raise RuntimeError(f"Negative token id in inputs: min={int(x.min())}")

    def forward(
        self,
        inputs: torch.Tensor,  # (batch_size, seq_len), ignored
        head_keys: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        masked_logits: (batch_size, max_action_dim)
                         Uniform logits (0.0) on valid actions; -1e5 on invalid.

        critic_vals:   (batch_size, 1) — always zero.
        """
        B = inputs.shape[0]

        # validate
        self._validate_inputs(inputs)

        # Default if none provided
        if head_keys is None:
            head_keys = [self.cmd_actor_head_key] * B

        # Raw logits: zeros everywhere -> uniform over all positions
        raw_logits = torch.zeros(B, self.MAX_ACTION_DIM, device=inputs.device)

        # Determine which mask‐row to use per batch element
        head_idx = torch.tensor([self.head2idx[k] for k in head_keys], device=inputs.device)

        # Apply each production's mask: masked_logits has shape (B, max_action_dim)
        keep_mask = self.all_action_masks[head_idx].eq(0)  # True where we keep logits
        masked_logits = torch.where(keep_mask, raw_logits, torch.full_like(raw_logits, self.MASK_VALUE))

        # Critic = zeros, not used
        critic_vals = torch.zeros(B, 1, device=inputs.device)

        return masked_logits, critic_vals
