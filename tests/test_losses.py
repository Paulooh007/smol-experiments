import torch

from src.losses import causal_lm_loss, load_balancing_loss, router_specialization_loss


def test_causal_lm_loss_masks_padding_positions():
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])

    loss_with_mask = causal_lm_loss(logits, labels, attention_mask)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :1, :].reshape(-1, 5),
        labels[:, 1:2].reshape(-1),
    )

    assert torch.allclose(loss_with_mask, expected)


def test_load_balancing_loss_is_finite_for_router_logits():
    router_logits = torch.tensor(
        [
            [[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]],
            [[0.0, 0.0, 2.0], [1.0, 2.0, 0.0]],
        ]
    )

    loss = load_balancing_loss(router_logits, num_experts=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_router_specialization_loss_ignores_padding_targets():
    router_logits = [
        torch.tensor(
            [
                [[8.0, 0.0, 0.0], [0.0, 0.0, 8.0], [2.0, 2.0, 2.0]],
            ]
        )
    ]
    target_experts = torch.tensor([[0, 2, -100]])

    loss = router_specialization_loss(router_logits, target_experts)

    assert loss.ndim == 0
    assert loss < 0.01
