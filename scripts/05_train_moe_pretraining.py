"""05 — Continued pre-training of the upcycled MoE on Cosmopedia-100k.

Right after upcycling, all experts are identical and the router is random, so
the MoE behaves exactly like the dense model. This stage "wakes up" the
mixture: a short run of standard next-token training plus a Switch-style load
balancing loss lets the experts drift apart while the router learns to spread
tokens across them (top-1 routing is unscaled, so the router only receives
gradients through the auxiliary loss — see src/models/moe.py).

Loss = causal_lm_loss + LB_LOSS_COEF * load_balancing_loss

Logged every LOG_EVERY steps: average train loss, validation loss, LB loss,
and active expert % (the canary for router collapse).

Crash-safe: training state is checkpointed every SAVE_EVERY steps; rerunning
the script resumes from the last checkpoint (data order and RNG restart, but
model/optimizer state is exact). The checkpoint is removed on completion.

Prerequisite: run scripts/04_upcycle_to_moe.py first.
Run from the repo root:  uv run python scripts/05_train_moe_pretraining.py
"""

import argparse
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from src.checkpoint import (
    load_moe_model,
    load_train_state,
    save_moe_model,
    save_train_state,
)
from src.data.pretraining import build_pretraining_dataloaders
from src.evaluation import compute_active_expert_pct, evaluate_moe_loss
from src.losses import causal_lm_loss
from src.metrics import CsvMetricLogger
from src.utils import amp_enabled, get_autocast_dtype, get_device, set_seed

# ----------------------------- Configuration ----------------------------------
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # tokenizer only; weights come from 04
INPUT_PATH = REPO_ROOT / "outputs" / "moe_upcycled.pt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "moe_pretrained.pt"
RESUME_PATH = REPO_ROOT / "outputs" / "moe_pretraining.resume.pt"
METRICS_PATH = REPO_ROOT / "outputs" / "moe_pretraining_metrics.csv"

SEED = 42
BLOCK_SIZE = 256          # sequence length of each training block
MAX_SAMPLES = 1000        # Cosmopedia documents to use (small demo run)
VAL_FRACTION = 0.2
BATCH_SIZE = 4            # fits easily in 16GB (T4) at 135M scale

STEPS = 100               # short demo run; raise for real continued pretraining
LEARNING_RATE = 3e-5      # gentle LR — we're nudging a converged model
LB_LOSS_COEF = 0.01       # weight on the load balancing auxiliary loss
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 10
SAVE_EVERY = 25           # training-state checkpoint interval (crash recovery)
# --------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--resume-path", type=Path, default=RESUME_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lb-loss-coef", type=float, default=LB_LOSS_COEF)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    set_seed(args.seed)
    device = get_device()
    amp_dtype = get_autocast_dtype()
    use_amp = amp_enabled(device)
    print(f"Device: {device} | autocast dtype: {amp_dtype} | AMP: {use_amp}")

    if not args.input_path.exists():
        sys.exit(
            f"Missing {args.input_path}. Run `uv run python scripts/04_upcycle_to_moe.py` first."
        )

    # Model weights stay float32; reduced precision comes from autocast. The
    # architecture comes from the checkpoint's bundled config, not defaults.
    model = load_moe_model(args.input_path)
    model.to(device).train()
    print(f"Loaded upcycled MoE from {args.input_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = build_pretraining_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        device=device,
        block_size=args.block_size,
        max_samples=args.max_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate / 10
    )
    # GradScaler is only needed for fp16 (bf16 and fp32 don't under/overflow).
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    start_step = 0
    if args.resume_path.exists():
        start_step, _ = load_train_state(
            args.resume_path, model, optimizer, scaler, scheduler
        )
        print(f"Resumed training state from {args.resume_path} at step {start_step}")

    train_iter = itertools.cycle(train_loader)  # loop the small dataset
    running_lm, running_lb = 0.0, 0.0
    metrics = CsvMetricLogger(
        args.metrics_path,
        [
            "phase",
            "step",
            "train_lm_loss",
            "val_loss",
            "lb_loss",
            "active_experts_pct",
            "lr",
        ],
    )

    with metrics:
        for step in range(start_step + 1, args.steps + 1):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                lm_loss = causal_lm_loss(outputs["logits"], input_ids, attention_mask)
                lb_loss = model.get_load_balancing_loss()
                loss = lm_loss + args.lb_loss_coef * lb_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip true (unscaled) gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_lm += lm_loss.item()
            running_lb += lb_loss.item()

            if step % args.log_every == 0:
                avg_lm = running_lm / args.log_every
                avg_lb = running_lb / args.log_every
                val_loss = evaluate_moe_loss(model, val_loader, device, amp_dtype)
                # Active expert % from a single validation batch.
                with torch.no_grad():
                    probe = next(iter(val_loader))
                    with torch.autocast(
                        device_type="cuda", dtype=amp_dtype, enabled=use_amp
                    ):
                        probe_out = model(
                            input_ids=probe["input_ids"].to(device),
                            attention_mask=probe["attention_mask"].to(device),
                        )
                    active_pct = compute_active_expert_pct(probe_out["router_logits"])
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step {step:>4}/{args.steps} | "
                    f"train loss {avg_lm:.4f} | "
                    f"val loss {val_loss:.4f} | "
                    f"LB loss {avg_lb:.4f} | "
                    f"active experts {active_pct:.1f}% | "
                    f"lr {lr:.2e}"
                )
                metrics.log(
                    phase="train",
                    step=step,
                    train_lm_loss=avg_lm,
                    val_loss=val_loss,
                    lb_loss=avg_lb,
                    active_experts_pct=active_pct,
                    lr=lr,
                )
                running_lm, running_lb = 0.0, 0.0
                model.train()

            if step % args.save_every == 0 and step < args.steps:
                save_train_state(args.resume_path, step, model, optimizer, scaler, scheduler)

    save_moe_model(model, args.output_path)  # weights bundled with config
    args.resume_path.unlink(missing_ok=True)
    print(f"\nSaved pre-trained MoE checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
