from pathlib import Path

import torch

from src.checkpoint import load_moe_model, save_moe_model
from src.models import SmolMoEConfig, SmolMoELM


def test_moe_checkpoint_round_trip_preserves_config_and_weights(tmp_path: Path):
    config = SmolMoEConfig(
        vocab_size=24,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=1,
        num_heads=3,
        kv_heads=1,
        num_experts=2,
    )
    model = SmolMoELM(config)
    path = tmp_path / "moe.pt"

    save_moe_model(model, path)
    loaded = load_moe_model(path)

    assert loaded.config == config
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
