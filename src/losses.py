"""Standalone loss functions used across the experiments.

All losses return scalar tensors and are composable, e.g.::

    loss = causal_lm_loss(logits, input_ids, attention_mask) \
         + 0.01 * model.get_load_balancing_loss() \
         + 0.01 * router_specialization_loss(router_logits, target_experts)
"""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Next-token cross-entropy with shifted logits/labels.

    Args:
        logits: [B, T, vocab_size]
        labels: [B, T] token ids (usually the input_ids themselves).
        attention_mask: optional [B, T]; positions with 0 (padding) are
            excluded from the loss. The notebooks trained on padding tokens —
            harmless for fully packed blocks, wrong for padded ones.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_mask == 0, IGNORE_INDEX)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer style auxiliary load balancing loss for one layer.

    loss = E * sum_e( load_e * importance_e )

    where ``load_e`` is the fraction of tokens routed (top-1) to expert e and
    ``importance_e`` is the mean router probability assigned to expert e
    (https://arxiv.org/abs/2101.03961). The loss equals 1.0 when routing is
    perfectly uniform and grows as routing collapses onto fewer experts.

    Args:
        router_logits: [B, T, num_experts] (pre-softmax gate outputs).
    """
    selected = router_logits.argmax(dim=-1)
    load = F.one_hot(selected, num_classes=num_experts).float().mean(dim=(0, 1))
    importance = F.softmax(router_logits.float(), dim=-1).mean(dim=(0, 1))
    return num_experts * torch.sum(load * importance)


def router_specialization_loss(
    all_router_logits: list[torch.Tensor],
    target_experts: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between router logits and per-token target experts.

    Treats each layer's router as a classifier over experts and supervises it
    directly with the expert that *should* handle each token (derived from the
    token's data domain). Averaged over layers; positions with target -100
    (padding) are ignored.

    Because routing in this repo is unscaled top-1, the LM loss sends no
    gradient to the routers — this loss is their only directional training
    signal, so its coefficient effectively just scales the router's learning
    rate rather than trading off against language modeling.

    Args:
        all_router_logits: per-layer [B, T, num_experts] router logits.
        target_experts: [B, T] expert index per token (-100 = ignore).
    """
    num_layers = len(all_router_logits)
    if num_layers == 0:
        return torch.zeros((), device=target_experts.device)
    total = torch.zeros((), device=all_router_logits[0].device)
    for router_logits in all_router_logits:
        total = total + F.cross_entropy(
            router_logits.reshape(-1, router_logits.shape[-1]).float(),
            target_experts.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    return total / num_layers
