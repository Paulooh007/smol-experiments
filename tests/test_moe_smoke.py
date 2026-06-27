import torch

from src.models import SmolMoEConfig, SmolMoELM


def tiny_config() -> SmolMoEConfig:
    return SmolMoEConfig(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=2,
        num_heads=3,
        kv_heads=1,
        num_experts=3,
        router_noise_std=0.0,
    )


def test_moe_forward_returns_logits_and_router_logits():
    model = SmolMoELM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["logits"].shape == (2, 5, model.config.vocab_size)
    assert len(outputs["router_logits"]) == model.config.num_hidden_layers
    assert outputs["router_logits"][0].shape == (2, 5, model.config.num_experts)


def test_moe_bookkeeping_after_forward():
    model = SmolMoELM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    model(input_ids=input_ids)
    lb_loss = model.get_load_balancing_loss()
    utilization = model.get_expert_utilization()

    assert lb_loss.ndim == 0
    assert torch.isfinite(lb_loss)
    assert utilization.shape == (
        model.config.num_hidden_layers,
        model.config.num_experts,
    )


def test_moe_greedy_generate_appends_tokens():
    model = SmolMoELM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (1, 3))

    generated = model.generate(input_ids, max_new_tokens=2)

    assert generated.shape == (1, 5)
