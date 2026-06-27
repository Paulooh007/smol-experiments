"""03 — Contrastive Preference Optimization (memory-efficient DPO alternative).

CPO (https://arxiv.org/abs/2401.08417) uses the same prompt/chosen/rejected
data as DPO but needs no separate reference model, so it roughly halves the
memory footprint and the per-step compute.

Pipeline:
    1. Load the SFT checkpoint from ./outputs/sft/ (CPO starts from SFT, not
       from the DPO model — the two are alternative branches to compare).
    2. Reuse the preference dataset saved by 02_train_dpo.py, or build it if
       it does not exist yet.
    3. Train with TRL's CPOTrainer, evaluate BLEU, save to ./outputs/cpo/.

Run from the repo root:  uv run python scripts/03_train_cpo.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import CPOConfig, CPOTrainer

from src.data.gec import load_gec_datasets
from src.data.preference import create_preference_dataset, generate_in_batches
from src.evaluation import evaluate_gec
from src.utils import get_autocast_dtype, get_device, set_seed

# ----------------------------- Hyperparameters --------------------------------
SFT_CHECKPOINT = REPO_ROOT / "outputs" / "sft"
OUTPUT_DIR = REPO_ROOT / "outputs" / "cpo"
PREFERENCE_DATASET_DIR = REPO_ROOT / "outputs" / "preference_dataset"

SEED = 42
MAX_PREFERENCE_SAMPLES = 4000   # only used if the saved dataset is missing
GENERATION_BATCH_SIZE = 64
VARIANT_1_PARAMS = {"max_new_tokens": 128, "do_sample": True, "temperature": 1.0}
VARIANT_2_PARAMS = {"max_new_tokens": 128, "do_sample": True, "temperature": 0.5}

BETA = 0.05
LEARNING_RATE = 5e-7
NUM_EPOCHS = 1
BATCH_SIZE = 8                  # no reference model -> larger batch fits
MAX_LENGTH = 350
MAX_PROMPT_LENGTH = 256
GENERATION_MAX_INPUT_LENGTH = 512
EVAL_MAX_INPUT_LENGTH = 512
LOGGING_STEPS = 100
# --------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-checkpoint", type=Path, default=SFT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--preference-dataset-dir", type=Path, default=PREFERENCE_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-preference-samples",
        type=int,
        default=MAX_PREFERENCE_SAMPLES,
        help="Number of CoEdIT train examples used if preferences are generated; set -1 for all.",
    )
    parser.add_argument("--generation-batch-size", type=int, default=GENERATION_BATCH_SIZE)
    parser.add_argument("--generation-max-input-length", type=int, default=GENERATION_MAX_INPUT_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--variant-1-temperature", type=float, default=1.0)
    parser.add_argument("--variant-2-temperature", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--eval-max-input-length", type=int, default=EVAL_MAX_INPUT_LENGTH)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=100)
    return parser.parse_args()


def build_or_load_preference_dataset(model, tokenizer, device, args: argparse.Namespace):
    """Load the dataset saved by script 02, building it only if necessary."""
    if args.preference_dataset_dir.exists():
        from datasets import load_from_disk

        print(f"Reusing preference dataset from {args.preference_dataset_dir}")
        return load_from_disk(str(args.preference_dataset_dir))

    print("No saved preference dataset found — generating one now.")
    max_samples = None if args.max_preference_samples < 0 else args.max_preference_samples
    train_ds, _ = load_gec_datasets(max_train_samples=max_samples)
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
    variants_1 = generate_in_batches(
        model,
        tokenizer,
        sources,
        device,
        variant_1_params,
        args.generation_batch_size,
        max_input_length=args.generation_max_input_length,
    )
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
    preference_ds.save_to_disk(str(args.preference_dataset_dir))
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

    cpo_config = CPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
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
    trainer = CPOTrainer(
        model=model,
        args=cpo_config,
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
    print(f"Saved CPO checkpoint to {args.output_dir}")

    print("Evaluating on the CoEdIT validation split (greedy decoding)...")
    _, test_ds = load_gec_datasets()
    scores = evaluate_gec(
        model,
        tokenizer,
        test_ds,
        device,
        max_input_length=args.eval_max_input_length,
    )
    # GLEU is the headline metric: unlike BLEU, it penalizes errors the model
    # left uncorrected (a copy-everything model scores high BLEU but low GLEU).
    print(f"SFT + CPO GLEU: {scores['gleu']:.4f} | BLEU: {scores['bleu']:.4f}")


if __name__ == "__main__":
    main()
