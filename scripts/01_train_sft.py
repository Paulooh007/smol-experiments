"""01 — Supervised fine-tuning of SmolLM-135M on grammatical error correction.

Pipeline:
    1. Load SmolLM-135M from HuggingFace.
    2. Load the Grammarly CoEdIT dataset, keep only the GEC task.
    3. Format examples as prompt/completion pairs and train with completion-only loss.
    4. Evaluate with BLEU on the held-out CoEdIT validation split.
    5. Save the checkpoint to ./outputs/sft/.

Run from the repo root:  uv run python scripts/01_train_sft.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from src.data.gec import (
    filter_by_prompt_token_length,
    filter_by_sft_token_length,
    format_for_sft_completion,
    load_gec_datasets,
)
from src.evaluation import evaluate_gec
from src.utils import get_autocast_dtype, get_device, set_seed

# ----------------------------- Hyperparameters ------------------------------
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
OUTPUT_DIR = REPO_ROOT / "outputs" / "sft"
EVAL_OUTPUT = REPO_ROOT / "outputs" / "sft_eval_predictions.csv"

SEED = 42
MAX_SEQ_LENGTH = 350      # CoEdIT GEC examples are short; 350 covers src+tgt
EVAL_MAX_INPUT_LENGTH = 512
MAX_TRAIN_INPUT_TOKENS = None
MAX_TRAIN_TOKENS = None
MAX_EVAL_INPUT_TOKENS = None
NUM_EPOCHS = 1            # ~0.48 BLEU after a single epoch
LEARNING_RATE = 7e-5      # tuned in the notebook; stable for a 135M model
BATCH_SIZE = 12           # comfortable on a 16GB T4 with packing at len 350
EVAL_FRACTION = 0.1       # in-training eval split carved from the train set
LOGGING_STEPS = 100
# Keep completion-only masking simple and correct by default. TRL can carry
# completion masks through packing, but it warns that packing without a
# supported FlashAttention implementation may cross-contaminate examples.
PACKING = False
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-output", type=Path, default=EVAL_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--eval-max-input-length", type=int, default=EVAL_MAX_INPUT_LENGTH)
    parser.add_argument(
        "--max-train-input-tokens",
        type=int,
        default=MAX_TRAIN_INPUT_TOKENS,
        help="Keep SFT train examples with formatted prompt at or below this token count.",
    )
    parser.add_argument(
        "--max-train-tokens",
        type=int,
        default=MAX_TRAIN_TOKENS,
        help="Keep SFT train examples with prompt+completion at or below this token count.",
    )
    parser.add_argument(
        "--max-eval-input-tokens",
        type=int,
        default=MAX_EVAL_INPUT_TOKENS,
        help="Keep validation examples with formatted prompt at or below this token count.",
    )
    parser.add_argument("--num-epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=PACKING)
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


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    set_seed(args.seed)
    device = get_device()
    amp_dtype = get_autocast_dtype()
    print(f"Device: {device} | autocast dtype: {amp_dtype}")

    # --- Data -----------------------------------------------------------------
    train_ds, test_ds = load_gec_datasets()
    print(f"GEC examples — train: {len(train_ds)}, test: {len(test_ds)}")

    # --- Model ------------------------------------------------------------------
    # padding_side="left" so batched generation (used in evaluation) is correct
    # for a decoder-only model. SmolLM has no pad token; reuse EOS.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    original_train_size = len(train_ds)
    original_test_size = len(test_ds)
    train_ds = filter_by_prompt_token_length(
        train_ds,
        tokenizer,
        max_tokens=args.max_train_input_tokens,
    )
    train_ds = filter_by_sft_token_length(
        train_ds,
        tokenizer,
        max_tokens=args.max_train_tokens,
        eos_token=tokenizer.eos_token,
    )
    test_ds = filter_by_prompt_token_length(
        test_ds,
        tokenizer,
        max_tokens=args.max_eval_input_tokens,
    )
    if (
        args.max_train_input_tokens is not None
        or args.max_train_tokens is not None
        or args.max_eval_input_tokens is not None
    ):
        print(
            "Length filter - "
            f"train: {len(train_ds)}/{original_train_size}, "
            f"test: {len(test_ds)}/{original_test_size}"
        )
    if len(train_ds) == 0:
        raise ValueError("No SFT training examples remain after token-length filtering.")
    if len(test_ds) == 0:
        raise ValueError("No validation examples remain after token-length filtering.")

    train_formatted = train_ds.map(
        lambda ex: format_for_sft_completion(ex, tokenizer.eos_token),
        remove_columns=list(train_ds.features),
        desc="Formatting prompt/completion pairs for SFT",
    )
    split = train_formatted.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    sft_train, sft_eval = split["train"], split["test"]

    # --- Train -------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_length=args.max_seq_length,
        packing=args.packing,
        completion_only_loss=True,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="steps",  # crash recovery; final model saved explicitly below
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=(device.type == "cuda" and amp_dtype == torch.bfloat16),
        fp16=(device.type == "cuda" and amp_dtype == torch.float16),
        report_to=[],
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_train,
        eval_dataset=sft_eval,
        processing_class=tokenizer,
    )
    # Resume from the last intermediate checkpoint if a previous run crashed.
    last_checkpoint = get_last_checkpoint(str(args.output_dir)) if args.output_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved SFT checkpoint to {args.output_dir}")

    # --- Evaluate ------------------------------------------------------------------
    print("Evaluating on the CoEdIT validation split (greedy decoding)...")
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
        f"SFT GLEU: {scores['gleu']:.4f} | BLEU: {scores['bleu']:.4f} | "
        f"copy {scores['exact_copy_rate']:.1%} | empty {scores['empty_rate']:.1%} | "
        f"edit→ref {scores['avg_edit_distance_to_reference']:.2f} "
        f"(BLEU expected ~0.48 after 1 epoch)"
    )
    print_length_bucket_scores(scores)
    print(f"Saved eval predictions to {args.eval_output}")


if __name__ == "__main__":
    main()
