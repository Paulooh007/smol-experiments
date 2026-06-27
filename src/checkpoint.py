"""Checkpointing for the custom MoE: config-bundled saves and resumable state.

Two kinds of files, both plain ``torch.save`` payloads of tensors/primitives:

1. **Model checkpoints** (scripts 04/05/06 outputs) bundle the model config
   with the weights, so downstream scripts reconstruct the *exact*
   architecture that was saved instead of assuming ``SmolMoEConfig()``
   defaults — editing the config in one script can no longer silently
   mismatch (or crash) the next one. Legacy bare state_dicts are still
   loadable for backward compatibility.

2. **Training-state checkpoints** additionally carry the optimizer, scaler,
   optional scheduler, torch RNG state and step counter, letting a training
   script resume after a crash or Colab timeout instead of restarting from
   step 0. Saves are atomic (write to a temp file, then rename) so an
   interruption mid-save can't corrupt an existing checkpoint. Resume is
   *near-exact*: model, optimizer and torch RNG streams (e.g. router noise)
   are restored precisely; only the DataLoader restarts from a fresh
   shuffle — acceptable for these experiments, noted here for honesty.
"""

from dataclasses import asdict
from pathlib import Path

import torch

from src.models import SmolMoEConfig, SmolMoELM


def save_moe_model(model: SmolMoELM, path: Path) -> None:
    """Save weights bundled with the config that defines their shapes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.config), "state_dict": model.state_dict()}, path
    )


def load_moe_model(path: Path, map_location: str = "cpu") -> SmolMoELM:
    """Rebuild a SmolMoELM from a checkpoint, using its bundled config.

    Falls back to ``SmolMoEConfig()`` defaults for legacy checkpoints that
    are bare state_dicts. ``load_state_dict`` is strict, so a genuine
    architecture mismatch fails loudly instead of loading garbage.
    """
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload and "config" in payload:
        config = SmolMoEConfig(**payload["config"])
        state_dict = payload["state_dict"]
    else:  # legacy format: the file is the state_dict itself
        config = SmolMoEConfig()
        state_dict = payload
    model = SmolMoELM(config)
    model.load_state_dict(state_dict)
    return model


def save_train_state(
    path: Path,
    step: int,
    model: SmolMoELM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler=None,
    **extra,
) -> None:
    """Atomically save everything needed to resume training at ``step``.

    ``extra`` keyword tensors/primitives (e.g. baseline eval results computed
    before training) are stored verbatim and returned by ``load_train_state``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload["rng"] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    payload["extra"] = extra
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on POSIX: never leaves a half-written file


def load_train_state(
    path: Path,
    model: SmolMoELM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler=None,
    map_location: str = "cpu",
) -> tuple[int, dict]:
    """Restore training state in place; returns (step, extra).

    The model must already be built with the same config (optimizer state
    tensors are moved to the parameters' device by ``load_state_dict``).
    """
    payload = torch.load(path, map_location=map_location)
    saved_config = SmolMoEConfig(**payload["config"])
    if saved_config != model.config:
        raise ValueError(
            f"Checkpoint config {saved_config} does not match model config "
            f"{model.config}; delete {path} to restart from scratch."
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    rng = payload.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload["step"], payload.get("extra", {})
