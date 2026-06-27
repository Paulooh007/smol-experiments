"""Sparse upcycling: convert a dense HF SmolLM checkpoint into a custom MoE.

Following https://arxiv.org/abs/2212.05055, every weight of the dense model
is reused: embeddings, attention, norms and the LM head are copied directly,
and the dense SwiGLU FFN is replicated into every expert of each MoE layer.
Only the router gates are freshly initialized.

Name mapping (HF Llama -> custom):
    q_proj/k_proj/v_proj/o_proj      -> W_query/W_key/W_value/W_output
    input_layernorm                  -> pre_attn_rmsnorm
    post_attention_layernorm         -> pre_moe_rmsnorm
    mlp.gate_proj/up_proj/down_proj  -> moe.gate_bank/up_bank/down_bank

Weight layout: HF ``nn.Linear`` stores [out_features, in_features] while the
expert banks store [num_experts, in, out], so FFN weights are transposed
before being replicated across the expert dimension.
"""

import math

import torch
from torch import nn

from src.models.moe import SmolMoELM


@torch.no_grad()
def upcycle_dense_to_moe(dense_model: nn.Module, moe_model: SmolMoELM) -> SmolMoELM:
    """Copy weights from a HF ``AutoModelForCausalLM`` SmolLM into ``moe_model``.

    Because the MoE uses unscaled top-1 routing and every expert starts as an
    exact copy of the dense FFN, the upcycled model's logits match the dense
    model's (up to float noise) regardless of what the random router picks.
    """
    # Token embeddings, LM head (tied in the dense checkpoint, so both copies
    # start identical here), and final norm.
    moe_model.model.embed_tokens.weight.copy_(dense_model.model.embed_tokens.weight)
    moe_model.lm_head.weight.copy_(dense_model.lm_head.weight)
    moe_model.model.norm.weight.copy_(dense_model.model.norm.weight)

    for dense_layer, moe_layer in zip(dense_model.model.layers, moe_model.model.layers):
        # Attention projections (same [out, in] layout on both sides).
        moe_layer.self_attn.W_query.weight.copy_(dense_layer.self_attn.q_proj.weight)
        moe_layer.self_attn.W_key.weight.copy_(dense_layer.self_attn.k_proj.weight)
        moe_layer.self_attn.W_value.weight.copy_(dense_layer.self_attn.v_proj.weight)
        moe_layer.self_attn.W_output.weight.copy_(dense_layer.self_attn.o_proj.weight)

        # Norms.
        moe_layer.pre_attn_rmsnorm.weight.copy_(dense_layer.input_layernorm.weight)
        moe_layer.pre_moe_rmsnorm.weight.copy_(dense_layer.post_attention_layernorm.weight)

        # Dense SwiGLU FFN -> expert banks: transpose [out, in] -> [in, out],
        # then replicate across the expert dimension.
        gate_t = dense_layer.mlp.gate_proj.weight.T
        up_t = dense_layer.mlp.up_proj.weight.T
        down_t = dense_layer.mlp.down_proj.weight.T
        moe_layer.moe.gate_bank.copy_(gate_t.unsqueeze(0).expand_as(moe_layer.moe.gate_bank))
        moe_layer.moe.up_bank.copy_(up_t.unsqueeze(0).expand_as(moe_layer.moe.up_bank))
        moe_layer.moe.down_bank.copy_(down_t.unsqueeze(0).expand_as(moe_layer.moe.down_bank))

        # Router: small random init — there is no dense counterpart to reuse.
        nn.init.kaiming_uniform_(moe_layer.moe.gate.weight, a=math.sqrt(5))

    return moe_model


@torch.no_grad()
def check_upcycling(
    dense_model: nn.Module,
    moe_model: SmolMoELM,
    input_ids: torch.Tensor,
    tolerance: float = 1e-3,
) -> float:
    """Compare dense vs. upcycled-MoE logits on ``input_ids``.

    Returns max |Δlogit| and raises ``AssertionError`` if it exceeds
    ``tolerance``. Run both models in float32 for a meaningful comparison.
    """
    dense_model.eval()
    moe_model.eval()
    attention_mask = torch.ones_like(input_ids)
    dense_logits = dense_model(input_ids=input_ids, attention_mask=attention_mask).logits
    moe_logits = moe_model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
    max_err = (dense_logits - moe_logits).abs().max().item()
    assert max_err < tolerance, (
        f"Upcycling sanity check failed: max |Δlogit| = {max_err:.6f} >= {tolerance}"
    )
    return max_err
