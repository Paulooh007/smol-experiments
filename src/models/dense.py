"""From-scratch dense SmolLM-135M (Llama-style decoder-only transformer)."""

from dataclasses import dataclass

import torch
from torch import nn

from src.models.components import (
    RMSNorm,
    RopeAttention,
    SwiGLUMLP,
    build_causal_mask,
)


@dataclass
class SmolConfig:
    """Hyperparameters of SmolLM-135M (HuggingFaceTB/SmolLM-135M)."""

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_heads: int = 9
    kv_heads: int = 3
    rope_theta: float = 10000.0


class DenseDecoderLayer(nn.Module):
    """Pre-norm decoder layer: RMSNorm -> attention -> residual, RMSNorm -> SwiGLU -> residual."""

    def __init__(self, config: SmolConfig):
        super().__init__()
        self.self_attn = RopeAttention(config)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.pre_mlp_rmsnorm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_mlp_rmsnorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class SmolModel(nn.Module):
    """Embeddings + N decoder layers + final norm (no LM head)."""

    def __init__(self, config: SmolConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DenseDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        # The causal mask is built once here (the notebooks rebuilt it inside
        # every layer) and combines causality with optional padding masking.
        causal_mask = build_causal_mask(
            seq_len=input_ids.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            padding_mask=attention_mask,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)
        return self.norm(hidden_states)


class SmolLM(nn.Module):
    """Full dense causal LM with tied input/output embeddings.

    SmolLM-135M ties the LM head to the token embeddings. We keep an explicit
    ``lm_head`` module for structural clarity (it mirrors the HF checkpoint
    layout and is what readers expect to find), but its weight *is* the
    embedding matrix — ``self.lm_head.weight`` aliases
    ``self.model.embed_tokens.weight``, so there is exactly one copy of the
    49152 x 576 matrix and the projection in ``forward`` is the tied-embedding
    projection. The original notebook instead kept an unused (dead) lm_head
    and did a manual ``matmul(h, embed.weight.T)``; weight tying achieves the
    same math without dead parameters.
    """

    def __init__(self, config: SmolConfig):
        super().__init__()
        self.config = config
        self.model = SmolModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """Returns {"logits": [B, T, vocab_size] float32 tensor}."""
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states).float()
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Simple greedy decoding (no KV cache; fine for small experiments)."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)["logits"]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if (
                eos_token_id is not None
                and input_ids.shape[0] == 1
                and next_token.item() == eos_token_id
            ):
                break
        return input_ids
