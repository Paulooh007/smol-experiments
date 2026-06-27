"""06 — Specialize MoE experts into Code, Math, and Chat domains.

Each token in the training stream carries a ``domain_id`` (0=Code, 1=Math,
2=Chat — see src/data/specialization.py). A per-token cross-entropy
"specialization loss" supervises every layer's router directly with the
expert that should handle the token's domain (Expert 0→Code, Expert 1→Math,
Expert 2→Chat). A small load balancing loss keeps routing from collapsing,
while the regular LM loss keeps training the experts on the tokens routed to
them.

    loss = causal_lm_loss
           + LB_LOSS_COEF * load_balancing_loss
           + SPEC_LOSS_COEF * router_specialization_loss

This is deliberately a single plain training loop — no staged freezing, no
per-group learning rates. Because routing here is unscaled top-1, the LM loss
sends *no* gradient to the routers; the router is trained by the load
balancing and specialization losses.

Functional evaluation: per-domain LM loss on held-out *documents* (the split
happens before chunking, so no document contributes blocks to both sides) is
printed BEFORE and AFTER training. Routing matrices (script 07) only show
that the router obeyed the specialization loss; this before/after comparison
shows whether specialization actually improved the model on each domain. For
a controlled comparison ("would plain training on the same data have done
just as well?"), set SPEC_LOSS_COEF = 0.0 and LB_LOSS_COEF = 0.0 and re-run:
same data, same steps, no router pressure.

Crash-safe: training state is checkpointed every SAVE_EVERY steps (including
the BEFORE baseline, so it isn't recomputed on a partially-trained model);
rerunning the script resumes from the last checkpoint. The checkpoint is
removed on completion.

Prerequisite: run scripts/05_train_moe_pretraining.py first.
Run from the repo root:  uv run python scripts/06_train_specialization.py
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
from src.data.specialization import (
    build_domain_val_loaders,
    build_specialization_dataloaders,
)
from src.evaluation import evaluate_per_domain_loss
from src.losses import causal_lm_loss, router_specialization_loss
from src.metrics import CsvMetricLogger
from src.utils import amp_enabled, get_autocast_dtype, get_device, set_seed

# ----------------------------- Configuration ----------------------------------
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # tokenizer only; weights come from 05
INPUT_PATH = REPO_ROOT / "outputs" / "moe_pretrained.pt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "moe_specialized.pt"
RESUME_PATH = REPO_ROOT / "outputs" / "moe_specialization.resume.pt"
METRICS_PATH = REPO_ROOT / "outputs" / "moe_specialization_metrics.csv"

SEED = 42
# Data settings: script 07 must use the SAME values to reproduce the same
# document-level train/val split (the split is deterministic in these
# arguments).
BLOCK_SIZE = 256
MAX_SAMPLES_PER_DOMAIN = 500   # Nemotron examples streamed per domain
VAL_FRACTION = 0.1
DATA_SEED = 100
BATCH_SIZE = 4

STEPS = 1500
LEARNING_RATE = 5e-5
LB_LOSS_COEF = 0.01
SPEC_LOSS_COEF = 0.01
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 50
SAVE_EVERY = 100       # training-state checkpoint interval (crash recovery)

# Domain id -> designated expert. With 3 experts and 3 domains this is the
# identity mapping, so per-token router targets are simply the domain_ids
# (padding stays -100 and is ignored by the loss).
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
    parser.add_argument("--max-samples-per-domain", type=int, default=MAX_SAMPLES_PER_DOMAIN)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lb-loss-coef", type=float, default=LB_LOSS_COEF)
    parser.add_argument("--spec-loss-coef", type=float, default=SPEC_LOSS_COEF)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    return parser.parse_args()


def print_domain_losses(title: str, losses: dict[str, float]) -> None:
    print(f"\n{title}")
    for name, value in losses.items():
        print(f"  {name.capitalize():<6} val loss: {value:.4f}")


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    set_seed(args.seed)
    device = get_device()
    amp_dtype = get_autocast_dtype()
    use_amp = amp_enabled(device)
    print(f"Device: {device} | autocast dtype: {amp_dtype} | AMP: {use_amp}")

    if not args.input_path.exists():
        sys.exit(
            f"Missing {args.input_path}. Run `uv run python scripts/05_train_moe_pretraining.py` first."
        )

    # The architecture comes from the checkpoint's bundled config, not defaults.
    model = load_moe_model(args.input_path)
    model.to(device)
    print(f"Loaded pre-trained MoE from {args.input_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, _, val_ds = build_specialization_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_samples_per_domain=args.max_samples_per_domain,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
    )
    domain_val_loaders = build_domain_val_loaders(val_ds, batch_size=args.batch_size)
    print(
        f"Train blocks: {len(train_loader.dataset)} | "
        f"held-out val blocks: {len(val_ds)} "
        f"({ {n: len(l.dataset) for n, l in domain_val_loaders.items()} })"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # GradScaler is only needed for fp16 (bf16 and fp32 don't under/overflow).
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    start_step = 0
    is_resuming = args.resume_path.exists()
    if args.resume_path.exists():
        start_step, extra = load_train_state(args.resume_path, model, optimizer, scaler)
        before = extra["before_losses"]  # baseline from the original run
        print(f"Resumed training state from {args.resume_path} at step {start_step}")
        print_domain_losses("Per-domain val loss BEFORE specialization (saved):", before)
    else:
        args.metrics_path.unlink(missing_ok=True)
        # Functional baseline: per-domain loss BEFORE specialization.
        before = evaluate_per_domain_loss(model, domain_val_loaders, device, amp_dtype)
        print_domain_losses("Per-domain val loss BEFORE specialization:", before)
    metrics = CsvMetricLogger(
        args.metrics_path,
        [
            "phase",
            "step",
            "domain",
            "lm_loss",
            "lb_loss",
            "spec_loss",
            "val_loss",
            "change",
        ],
    )
    for name, value in before.items():
        if not is_resuming:
            metrics.log(phase="before", step=start_step, domain=name, val_loss=value)

    model.train()
    train_iter = itertools.cycle(train_loader)
    running = {"lm": 0.0, "lb": 0.0, "spec": 0.0}

    with metrics:
        for step in range(start_step + 1, args.steps + 1):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_experts = batch["domain_ids"].to(device)  # identity map; pads -100

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                lm_loss = causal_lm_loss(outputs["logits"], input_ids, attention_mask)
                spec_loss = router_specialization_loss(
                    outputs["router_logits"], target_experts
                )
                lb_loss = model.get_load_balancing_loss()
                loss = (
                    lm_loss
                    + args.lb_loss_coef * lb_loss
                    + args.spec_loss_coef * spec_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip true (unscaled) gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running["lm"] += lm_loss.item()
            running["lb"] += lb_loss.item()
            running["spec"] += spec_loss.item()

            if step % args.log_every == 0:
                avg_lm = running["lm"] / args.log_every
                avg_lb = running["lb"] / args.log_every
                avg_spec = running["spec"] / args.log_every
                print(
                    f"step {step:>4}/{args.steps} | "
                    f"lm loss {avg_lm:.4f} | "
                    f"LB loss {avg_lb:.4f} | "
                    f"spec loss {avg_spec:.4f}"
                )
                metrics.log(
                    phase="train",
                    step=step,
                    lm_loss=avg_lm,
                    lb_loss=avg_lb,
                    spec_loss=avg_spec,
                )
                running = {"lm": 0.0, "lb": 0.0, "spec": 0.0}

            if step % args.save_every == 0 and step < args.steps:
                save_train_state(
                    args.resume_path, step, model, optimizer, scaler, before_losses=before
                )

        # --- Functional result: per-domain loss AFTER specialization ------------
        after = evaluate_per_domain_loss(model, domain_val_loaders, device, amp_dtype)
        print_domain_losses("Per-domain val loss AFTER specialization:", after)
        print("\nChange (negative = improvement):")
        for name in before:
            change = after[name] - before[name]
            print(f"  {name.capitalize():<6} {change:+.4f}")
            metrics.log(
                phase="after",
                step=args.steps,
                domain=name,
                val_loss=after[name],
                change=change,
            )

    save_moe_model(model, args.output_path)  # weights bundled with config
    args.resume_path.unlink(missing_ok=True)
    print(f"\nSaved specialized MoE checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
