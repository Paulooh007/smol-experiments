"""Building blocks shared by the dense and MoE SmolLM implementations.

These mirror the Llama-style architecture used by SmolLM-135M:
rotary position embeddings, RMSNorm, SwiGLU MLP, and grouped-query
attention (9 query heads sharing 3 KV heads).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping halves and negating: (a, b) -> (-b, a)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Apply rotary embeddings to query/key tensors of shape [B, H, T, D_head]."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for grouped-query attention.

    [B, kv_heads, T, D_head] -> [B, kv_heads * n_rep, T, D_head]
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def build_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the additive attention mask used by every decoder layer.

    Returns a [1, 1, T, T] (or [B, 1, T, T] when ``padding_mask`` is given)
    tensor of zeros at allowed positions and a large negative value at
    disallowed positions. The large-but-finite ``finfo(dtype).min`` is used
    instead of ``-inf`` so a fully masked row softmaxes to a uniform
    distribution rather than NaN.

    Args:
        seq_len: Sequence length T.
        device: Device the mask should live on.
        dtype: Dtype of the attention scores (the mask is added to them).
        padding_mask: Optional [B, T] tensor with 1 for real tokens and 0 for
            padding. Padded *keys* are masked out for every query. Sequences
            are assumed to be right-padded.
    """
    min_val = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype)
    mask = mask.triu(diagonal=1)[None, None, :, :]
    if padding_mask is not None:
        pad = (1.0 - padding_mask[:, None, None, :].to(dtype)) * min_val
        # clamp: adding two min_vals would overflow to -inf
        mask = (mask + pad).clamp_min(min_val)
    return mask


class RotaryEmbedder(nn.Module):
    """Computes RoPE cos/sin tables on the device and dtype of the input.

    The inverse frequencies are registered as a non-persistent buffer so they
    follow ``model.to(device)`` automatically (the original notebook stored
    them as a plain attribute, which silently stayed on CPU and broke on GPU).
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """x is any tensor shaped [..., T, D]; only T, device and dtype are used."""
        seq_len = x.shape[-2]
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        angles = torch.outer(pos, self.inv_freq).unsqueeze(0)  # [1, T, dim/2]
        emb = torch.cat((angles, angles), dim=-1)  # [1, T, dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class RMSNorm(nn.Module):
    """Root-mean-square normalization (no mean centering, no bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for numerical stability under fp16/bf16.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward block: down(SiLU(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class RopeAttention(nn.Module):
    """Multi-head attention with RoPE and grouped-query attention.

    SmolLM-135M uses 9 query heads and 3 KV heads; KV heads are expanded
    with :func:`repeat_kv` before the attention matmul.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.kv_heads = config.kv_heads
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        self.W_query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.W_key = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_value = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedder(dim=self.head_dim, base=self.rope_theta)

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Args:
            hidden_states: [B, T, hidden_size]
            causal_mask: additive mask from :func:`build_causal_mask`,
                broadcastable to [B, num_heads, T, T].
        """
        b, t, _ = hidden_states.size()

        q_states = self.W_query(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = self.W_key(hidden_states).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v_states = self.W_value(hidden_states).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v_states)
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        kv_groups = self.num_heads // self.kv_heads
        k_states = repeat_kv(k_states, kv_groups)
        v_states = repeat_kv(v_states, kv_groups)

        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(b, t, -1)
        return self.W_output(attn_output)
