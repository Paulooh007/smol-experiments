"""02 — Direct Preference Optimization on top of the SFT checkpoint.

Pipeline:
    1. Load the SFT checkpoint from ./outputs/sft/.
    2. For each GEC training example, sample two corrections (temperature 1.0
       and 0.5). Rank them by edit distance to the ground truth: closer =
       chosen, farther = rejected; ties are dropped.
    3. Train with TRL's DPOTrainer (a frozen reference copy of the policy is
       created automatically when ref_model=None).
    4. Evaluate BLEU and save to ./outputs/dpo/.

The generated preference dataset is also saved to ./outputs/preference_dataset
so 03_train_cpo.py can reuse it without regenerating.

Run from the repo root:  uv run python scripts/02_train_dpo.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

from src.data.gec import filter_by_prompt_token_length, load_gec_datasets
from src.data.preference import create_preference_dataset, generate_in_batches
from src.evaluation import evaluate_gec
from src.utils import get_autocast_dtype, get_device, set_seed

# ----------------------------- Hyperparameters --------------------------------
SFT_CHECKPOINT = REPO_ROOT / "outputs" / "sft"
OUTPUT_DIR = REPO_ROOT / "outputs" / "dpo"
PREFERENCE_DATASET_DIR = REPO_ROOT / "outputs" / "preference_dataset"
EVAL_OUTPUT = REPO_ROOT / "outputs" / "dpo_eval_predictions.csv"

SEED = 42
# Generating two samples for all ~20k train examples is the slowest part of
# this script. 4000 prompts already yields a few thousand non-tie pairs;
# raise this (up to None for the full set) if you have GPU time to spare.
MAX_PREFERENCE_SAMPLES = 4000
GENERATION_BATCH_SIZE = 64          # halved automatically on CUDA OOM
VARIANT_1_PARAMS = {"max_new_tokens": 128, "do_sample": True, "temperature": 1.0}
VARIANT_2_PARAMS = {"max_new_tokens": 128, "do_sample": True, "temperature": 0.5}

BETA = 0.05               # KL penalty strength; small beta = stronger updates
LEARNING_RATE = 5e-7      # DPO is sensitive — keep the LR tiny
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
MAX_LENGTH = 350
MAX_PROMPT_LENGTH = 256
GENERATION_MAX_INPUT_LENGTH = 512
EVAL_MAX_INPUT_LENGTH = 512
MAX_PREFERENCE_INPUT_TOKENS = None
MAX_EVAL_INPUT_TOKENS = None
LOGGING_STEPS = 100
# --------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-checkpoint", type=Path, default=SFT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--preference-dataset-dir", type=Path, default=PREFERENCE_DATASET_DIR)
    parser.add_argument("--eval-output", type=Path, default=EVAL_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-preference-samples",
        type=int,
        default=MAX_PREFERENCE_SAMPLES,
        help="Number of CoEdIT train examples used to build preferences; set -1 for all.",
    )
    parser.add_argument("--generation-batch-size", type=int, default=GENERATION_BATCH_SIZE)
    parser.add_argument("--generation-max-input-length", type=int, default=GENERATION_MAX_INPUT_LENGTH)
    parser.add_argument(
        "--max-preference-input-tokens",
        type=int,
        default=MAX_PREFERENCE_INPUT_TOKENS,
        help="Build DPO preference pairs only from prompts at or below this token count.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--variant-1-temperature", type=float, default=1.0)
    parser.add_argument("--variant-2-temperature", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--eval-max-input-length", type=int, default=EVAL_MAX_INPUT_LENGTH)
    parser.add_argument(
        "--max-eval-input-tokens",
        type=int,
        default=MAX_EVAL_INPUT_TOKENS,
        help="Keep validation examples with formatted prompt at or below this token count.",
    )
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--rebuild-preference-dataset",
        action="store_true",
        help="Delete and regenerate the saved preference dataset if it already exists.",
    )
    return parser.parse_args()


def print_length_bucket_scores(scores: dict) -> None:
    buckets = scores.get("length_buckets")
    if not buckets:
        return
    print("Length buckets (prompt tokens):")
    for name in ("short", "medium", "long"):
        bucket = buckets.get(name)
        if not bucket or bucket["n"] == 0:
            continue
        print(
            f"  {name:<6} n={bucket['n']:>4} | "
            f"GLEU {bucket['gleu']:.4f} | BLEU {bucket['bleu']:.4f} | "
            f"copy {bucket['exact_copy_rate']:.1%} | "
            f"edit-ref {bucket['avg_edit_distance_to_reference']:.2f}"
        )


def preference_dataset_metadata(args: argparse.Namespace) -> dict:
    return {
        "max_preference_samples": args.max_preference_samples,
        "max_preference_input_tokens": args.max_preference_input_tokens,
        "generation_max_input_length": args.generation_max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "variant_1_temperature": args.variant_1_temperature,
        "variant_2_temperature": args.variant_2_temperature,
    }


def build_or_load_preference_dataset(model, tokenizer, device, args: argparse.Namespace):
    """Generate two variants per prompt and rank them by edit distance."""
    if args.preference_dataset_dir.exists():
        from datasets import load_from_disk

        metadata_path = args.preference_dataset_dir / "metadata.json"
        if args.rebuild_preference_dataset:
            print(f"Rebuilding preference dataset at {args.preference_dataset_dir}")
            shutil.rmtree(args.preference_dataset_dir)
        elif metadata_path.exists():
            expected = preference_dataset_metadata(args)
            actual = json.loads(metadata_path.read_text())
            if actual != expected:
                raise ValueError(
                    "Existing preference dataset metadata does not match this run. "
                    "Pass --rebuild-preference-dataset or use a new "
                    "--preference-dataset-dir."
                )
            print(f"Reusing preference dataset from {args.preference_dataset_dir}")
            return load_from_disk(str(args.preference_dataset_dir))
        elif args.max_preference_input_tokens is not None:
            raise ValueError(
                "Existing preference dataset has no metadata, so it cannot be "
                "safely reused with --max-preference-input-tokens. Pass "
                "--rebuild-preference-dataset or use a new --preference-dataset-dir."
            )
        else:
            print(f"Reusing legacy preference dataset from {args.preference_dataset_dir}")
            return load_from_disk(str(args.preference_dataset_dir))

    max_samples = None if args.max_preference_samples < 0 else args.max_preference_samples
    train_ds, _ = load_gec_datasets()
    original_train_size = len(train_ds)
    train_ds = filter_by_prompt_token_length(
        train_ds,
        tokenizer,
        max_tokens=args.max_preference_input_tokens,
    )
    filtered_train_size = len(train_ds)
    if max_samples is not None:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
    if len(train_ds) == 0:
        raise ValueError("No preference examples remain after token-length filtering.")
    if args.max_preference_input_tokens is not None:
        print(
            "Preference length filter - "
            f"eligible: {filtered_train_size}/{original_train_size}, "
            f"using: {len(train_ds)}"
        )
    sources = list(train_ds["src"])
    targets = list(train_ds["tgt"])
    variant_1_params = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.variant_1_temperature,
    }
    variant_2_params = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.variant_2_temperature,
    }

    print(f"Generating variant 1 (T={variant_1_params['temperature']}) for {len(sources)} prompts...")
    variants_1 = generate_in_batches(
        model,
        tokenizer,
        sources,
        device,
        variant_1_params,
        args.generation_batch_size,
        max_input_length=args.generation_max_input_length,
    )
    print(f"Generating variant 2 (T={variant_2_params['temperature']})...")
    variants_2 = generate_in_batches(
        model,
        tokenizer,
        sources,
        device,
        variant_2_params,
        args.generation_batch_size,
        max_input_length=args.generation_max_input_length,
    )

    preference_ds = create_preference_dataset(sources, targets, variants_1, variants_2)
    print(f"Preference pairs (ties dropped): {len(preference_ds)}")
    preference_ds.save_to_disk(str(args.preference_dataset_dir))
    metadata_path = args.preference_dataset_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(preference_dataset_metadata(args), indent=2, sort_keys=True)
    )
    return preference_ds


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    set_seed(args.seed)
    device = get_device()
    amp_dtype = get_autocast_dtype()
    print(f"Device: {device} | autocast dtype: {amp_dtype}")

    if not args.sft_checkpoint.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found at {args.sft_checkpoint}. Run scripts/01_train_sft.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(args.sft_checkpoint), padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(args.sft_checkpoint)).to(device)

    preference_ds = build_or_load_preference_dataset(model, tokenizer, device, args)

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
        loss_type="sigmoid",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        lr_scheduler_type="cosine",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_strategy="steps",  # crash recovery; final model saved explicitly below
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=(device.type == "cuda" and amp_dtype == torch.bfloat16),
        fp16=(device.type == "cuda" and amp_dtype == torch.float16),
        report_to=[],
        seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer clones the policy as the frozen reference
        args=dpo_config,
        train_dataset=preference_ds,
        processing_class=tokenizer,
    )
    # Resume from the last intermediate checkpoint if a previous run crashed.
    last_checkpoint = get_last_checkpoint(str(args.output_dir)) if args.output_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved DPO checkpoint to {args.output_dir}")

    print("Evaluating on the CoEdIT validation split (greedy decoding)...")
    _, test_ds = load_gec_datasets()
    original_test_size = len(test_ds)
    test_ds = filter_by_prompt_token_length(
        test_ds,
        tokenizer,
        max_tokens=args.max_eval_input_tokens,
    )
    if args.max_eval_input_tokens is not None:
        print(f"Eval length filter - test: {len(test_ds)}/{original_test_size}")
    if len(test_ds) == 0:
        raise ValueError("No validation examples remain after token-length filtering.")
    scores = evaluate_gec(
        model,
        tokenizer,
        test_ds,
        device,
        max_input_length=args.eval_max_input_length,
        output_path=args.eval_output,
    )
    # GLEU is the headline metric: unlike BLEU, it penalizes errors the model
    # left uncorrected (a copy-everything model scores high BLEU but low GLEU).
    print(
        f"SFT + DPO GLEU: {scores['gleu']:.4f} | BLEU: {scores['bleu']:.4f} | "
        f"copy {scores['exact_copy_rate']:.1%} | empty {scores['empty_rate']:.1%} | "
        f"edit→ref {scores['avg_edit_distance_to_reference']:.2f} "
        f"(BLEU expected ~0.50)"
    )
    print_length_bucket_scores(scores)
    print(f"Saved eval predictions to {args.eval_output}")


if __name__ == "__main__":
    main()
