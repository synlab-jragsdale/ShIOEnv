import torch
from torch import nn
from typing import Tuple, List, Dict, Optional
import math


class PolicyNetwork(nn.Module):
    """
    policy network with shared actor-critic trunk
    The network branches only at the heads.
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

        # hyperparams
        self.embed_dim = config.get("embed_dim", 128)
        self.hidden_dim = config.get("h_dim", 512)
        self.n_heads = config.get("n_heads", 4)
        self.n_layers = config.get("n_layers", 4)
        self.max_seq_len = config.get("input_size", 64)  # used for positional embeddings
        self.cls_token_id = 1

        self.cmd_actor_head_key = cmd_key

        if self.embed_dim % self.n_heads:
            raise ValueError("embed_dim must be divisible by n_heads")

        self.head2idx = {k: i for i, k in enumerate(production_masks.keys())}

        self.MAX_ACTION_DIM = max_action_dim
        self.MASK_VALUE = -1e5

        # Build one giant mask tensor: 0 keeps a logit, -1e5 blanks it out when softmaxed by torch.Categorical
        all_masks = torch.full(
            (len(production_masks), self.MAX_ACTION_DIM),
            fill_value=self.MASK_VALUE,  # default = “masked out”
            dtype=torch.float32,
        )

        for k, mask in production_masks.items():
            row = self.head2idx[k]  # integer row for that production
            keep = mask.to(torch.bool)  # 1->keep, 0->mask
            all_masks[row, keep] = 0.0  # zero means “leave the logit as‑is”

        # Put on the same device as the model but mark it non‑persistent
        self.register_buffer("all_action_masks", all_masks, persistent=False)

        # shared trunk
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

        # Pre‑allocate position indices to avoid realloc each fwd
        self.register_buffer("_pos_ids", torch.arange(self.max_seq_len).unsqueeze(0), persistent=False)

        trunk_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.0,
            batch_first=True   # (seq_len, batch, dim)
        )
        self.transformer = nn.TransformerEncoder(
            trunk_layer, num_layers=self.n_layers, norm=None
        )

        # heads
        # Actor: produces logits for all productions then mask
        self.action_head = nn.Linear(self.embed_dim, self.MAX_ACTION_DIM)

        # Critic: scalar value function
        self.critic_head = nn.Linear(self.embed_dim, 1)

        self.init_weights()

    def init_weights(self):
        # embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        # every linear except last heads
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in {self.action_head, self.critic_head}:
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # actor head: near‑uniform logits
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)

        # critic head
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    @torch.no_grad()
    def _validate_inputs(self, x: torch.Tensor):
        if torch.any(x < 0) or torch.any(x >= self.embedding.num_embeddings):
            raise RuntimeError(
                f"Token id out of range: min={int(x.min())}, max={int(x.max())}, "
                f"vocab={self.embedding.num_embeddings}"
            )

    def forward(
            self,
            inputs: torch.Tensor,  # (batch_size, seq_len)
            head_keys: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          masked_logits:  (batch_size, max_action_dim)
                          For each batch element i, valid positions are determined
                          by production_masks[head_keys[i]].
          critic_vals:    (batch_size, 1)
        """
        self._validate_inputs(inputs)
        B, S = inputs.shape
        # Default if not provided
        if head_keys is None:
            head_keys = [self.cmd_actor_head_key] * B

        # Insert [CLS] at front, remove the last token => shape remains (B, seq_len)
        cls_col = torch.full((B, 1), self.cls_token_id, dtype=torch.long, device=inputs.device)
        inputs_with_cls = torch.cat([cls_col, inputs[:, :-1]], dim=1)

        # Embeddings
        emb = self.embedding(inputs_with_cls)  # (B, seq_len, embed_dim)

        # Positional embeddings
        pos = self.pos_embedding(self._pos_ids[:, :S])
        h = emb + pos

        # Key padding mask (True -> pad). Using '0' as pad_idx
        key_padding = inputs_with_cls.eq(0)  # (B, S)

        # Run through Trans
        trans_out = self.transformer(h, src_key_padding_mask=key_padding)

        # Gather final hidden states
        hidden = trans_out[:, 0, :]  # shape (B, embed_dim)

        # Single head -> raw actions for entire batch (B, max_action_dim)
        raw_logits = self.action_head(hidden)

        # Apply each production's mask: masked_logits has shape (B, max_action_dim)
        head_idx = torch.as_tensor([self.head2idx[k] for k in head_keys], device=raw_logits.device)
        keep_mask = self.all_action_masks[head_idx].eq(0)
        masked_logits = torch.where(keep_mask, raw_logits, torch.tensor(self.MASK_VALUE, device=raw_logits.device))

        # critic values
        critic_values = self.critic_head(hidden)  # (B, 1)

        return masked_logits, critic_values
