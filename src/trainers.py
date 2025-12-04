import torch
import torch.nn.functional as F

def compute_router_guidance_loss(all_router_logits, domain_ids, config):
    """
    Calculates KL Divergence loss to force router specialization.
    """
    total_loss = 0
    num_layers = len(all_router_logits)
    device = all_router_logits[0].device

    num_domains = len(config.expert_map)

    # Target distribution: High prob for assigned expert, low prob for others
    target_distributions = torch.full((num_domains, config.num_experts), config.low_p, device=device, dtype=torch.bfloat16)
    for domain_id, expert_ids in config.expert_map.items():
        target_distributions[domain_id, expert_ids] = config.high_p

    for router_logits in all_router_logits:
        B, T, E = router_logits.shape
        router_logits_flat = router_logits.view(-1, E)
        domain_ids_flat = domain_ids.view(-1)

        valid_mask = (domain_ids_flat >= 0) & (domain_ids_flat < num_domains)
        
        valid_logits = router_logits_flat[valid_mask]
        valid_domains = domain_ids_flat[valid_mask]

        if valid_logits.shape[0] == 0:
            continue

        router_log_probs = F.log_softmax(valid_logits, dim=-1)
        targets = target_distributions[valid_domains]

        layer_loss = F.kl_div(router_log_probs, targets, log_target=False, reduction='batchmean')
        
        if torch.isfinite(layer_loss):
            total_loss += layer_loss

    if num_layers == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / num_layers

def causal_lm_loss(logits, labels):
    """Standard Causal Language Modeling Loss"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss