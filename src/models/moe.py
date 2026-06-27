"""From-scratch Mixture-of-Experts SmolLM.

This is the single, unified MoE implementation (the source notebook defined
the model twice with diverging behavior). The model *always* returns both LM
logits and per-layer router logits:

- the continued pre-training script simply ignores ``router_logits``;
- the specialization script feeds them to the router specialization loss.

The router gate is computed exactly once per layer per forward pass —
``MoELayer.forward`` returns ``(output, router_logits)`` (the notebook's
specialization variant ran the gate twice).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.losses import load_balancing_loss
from src.models.components import RMSNorm, RopeAttention, build_causal_mask


@dataclass
class SmolMoEConfig:
    """SmolLM-135M hyperparameters plus MoE settings."""

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_heads: int = 9
    kv_heads: int = 3
    rope_theta: float = 10000.0
    num_experts: int = 3
    num_experts_per_tok: int = 1  # only top-1 routing is implemented
    # Gaussian noise added to router logits during training. Encourages
    # exploration so the freshly initialized router doesn't collapse onto a
    # single expert immediately after upcycling.
    router_noise_std: float = 0.1


class MoELayer(nn.Module):
    """Top-1 routed mixture of SwiGLU experts stored as 3D parameter banks.

    Expert weights live in banks shaped [E, in, out] so all experts can be
    evaluated with a single einsum. Every expert is computed for every token
    and the routed output is then gathered — wasteful FLOPs-wise but simple,
    vectorized, and perfectly adequate at 135M scale (E=3).

    Two deliberate simplifications, kept from the notebook:

    - Routing is hard top-1 (argmax) and the selected expert's output is NOT
      scaled by its router probability (Switch Transformers scale by it).
      This keeps upcycled MoE outputs *bit-equivalent* to the dense model and
      means the router receives gradients only through auxiliary losses (load
      balancing / specialization), not through the LM loss.
    - Parameter banks default to float32; scripts that want reduced precision
      cast the whole model after construction (``model.to(dtype)``) rather
      than mixing dtypes inside the model.
    """

    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        if config.num_experts_per_tok != 1:
            raise NotImplementedError(
                "Only top-1 routing (num_experts_per_tok=1) is implemented."
            )
        self.num_experts = config.num_experts
        self.emb_dim = config.hidden_size
        self.moe_dim = config.intermediate_size
        self.router_noise_std = config.router_noise_std

        self.gate = nn.Linear(self.emb_dim, self.num_experts, bias=False)
        self.gate_bank = nn.Parameter(torch.empty(self.num_experts, self.emb_dim, self.moe_dim))
        self.up_bank = nn.Parameter(torch.empty(self.num_experts, self.emb_dim, self.moe_dim))
        self.down_bank = nn.Parameter(torch.empty(self.num_experts, self.moe_dim, self.emb_dim))
        self.reset_parameters()

        # Populated on every forward; consumed by SmolMoELM helper methods.
        self._last_lb_loss: torch.Tensor | None = None
        self._last_tokens_per_expert: torch.Tensor | None = None

    def reset_parameters(self) -> None:
        """Kaiming init for the banks (the notebook left them uninitialized)."""
        for bank in (self.gate_bank, self.up_bank, self.down_bank):
            nn.init.kaiming_uniform_(bank, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: [B, T, hidden_size]

        Returns:
            output: [B, T, hidden_size] — the selected expert's output per token.
            router_logits: [B, T, num_experts] — gate logits (post training
                noise), computed exactly once.
        """
        router_logits = self.gate(x)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std

        selected = router_logits.argmax(dim=-1)  # [B, T]

        # Evaluate all experts at once, then gather the routed one.
        a = torch.einsum("btd,edh->bteh", x, self.gate_bank)
        u = torch.einsum("btd,edh->bteh", x, self.up_bank)
        h = F.silu(a) * u
        y_all = torch.einsum("bteh,ehd->bted", h, self.down_bank)

        gather_idx = selected[..., None, None].expand(-1, -1, 1, self.emb_dim)
        output = torch.gather(y_all, dim=2, index=gather_idx).squeeze(2)

        # Bookkeeping for get_load_balancing_loss / get_expert_utilization.
        self._last_lb_loss = load_balancing_loss(router_logits, self.num_experts)
        with torch.no_grad():
            self._last_tokens_per_expert = torch.bincount(
                selected.reshape(-1), minlength=self.num_experts
            )

        return output, router_logits


class MoEDecoderLayer(nn.Module):
    """Pre-norm decoder layer with an MoE FFN: attention -> MoE with residuals."""

    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        self.self_attn = RopeAttention(config)
        self.moe = MoELayer(config)
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.pre_moe_rmsnorm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self, hidden_states: torch.Tensor, causal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_moe_rmsnorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        return hidden_states + residual, router_logits


class SmolMoEModel(nn.Module):
    """Embeddings + N MoE decoder layers + final norm."""

    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MoEDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.embed_tokens(input_ids)
        causal_mask = build_causal_mask(
            seq_len=input_ids.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            padding_mask=attention_mask,
        )
        all_router_logits = []
        for layer in self.layers:
            hidden_states, router_logits = layer(hidden_states, causal_mask)
            all_router_logits.append(router_logits)
        return self.norm(hidden_states), all_router_logits


class SmolMoELM(nn.Module):
    """Full MoE causal LM.

    ``forward`` always returns ``{"logits": ..., "router_logits": [...]}``;
    callers that don't need router logits simply ignore them.

    Unlike the dense model, the LM head is a separate (untied) parameter: it
    is initialized from the dense checkpoint during upcycling and is then
    free to diverge from the embeddings during continued training.
    """

    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        self.config = config
        self.model = SmolMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        hidden_states, all_router_logits = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states).float()
        return {"logits": logits, "router_logits": all_router_logits}

    def get_load_balancing_loss(self) -> torch.Tensor:
        """Mean Switch-style load balancing loss across layers (differentiable).

        Uses values stashed during the most recent forward pass; call after
        ``forward`` within the same autograd graph.
        """
        losses = [layer.moe._last_lb_loss for layer in self.model.layers]
        if any(loss is None for loss in losses):
            raise RuntimeError(
                "get_load_balancing_loss() called before any forward pass."
            )
        return torch.stack(losses).mean()

    @torch.no_grad()
    def get_expert_utilization(self) -> torch.Tensor:
        """Token counts per expert from the last forward: [num_layers, num_experts]."""
        counts = [layer.moe._last_tokens_per_expert for layer in self.model.layers]
        if any(c is None for c in counts):
            raise RuntimeError(
                "get_expert_utilization() called before any forward pass."
            )
        return torch.stack(counts)

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
