"""04 — Upcycle dense SmolLM-135M into a 3-expert Mixture-of-Experts.

Following "Sparse Upcycling" (https://arxiv.org/abs/2212.05055):
    1. Load dense SmolLM-135M from HuggingFace in float32.
    2. Build the custom SmolMoELM (3 experts, top-1 routing).
    3. Copy embeddings, attention, norms and LM head; transpose and replicate
       the dense SwiGLU FFN into each expert bank; randomly init the routers.
    4. Sanity check: max |Δlogit| between dense and MoE must be < 1e-3
       (because routing is unscaled top-1 and all experts start identical,
       the upcycled model is numerically equivalent to the dense one).
    5. Save the MoE weights (bundled with their config) to
       ./outputs/moe_upcycled.pt.

Runs comfortably on CPU — no training happens here.

Run from the repo root:  uv run python scripts/04_upcycle_to_moe.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import SmolMoEConfig, SmolMoELM
from src.checkpoint import save_moe_model
from src.upcycling import check_upcycling, upcycle_dense_to_moe
from src.utils import count_parameters, get_device, set_seed

# ----------------------------- Configuration ----------------------------------
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
OUTPUT_PATH = REPO_ROOT / "outputs" / "moe_upcycled.pt"

SEED = 42
PARITY_TOLERANCE = 1e-3
TEST_PROMPT = "Where is the Great Wall?"
NUM_TEST_TOKENS = 30  # length of the greedy generation shown for both models
# The MoE shape (3 experts, top-1) comes from SmolMoEConfig defaults.
# --------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--parity-tolerance", type=float, default=PARITY_TOLERANCE)
    parser.add_argument("--test-prompt", default=TEST_PROMPT)
    parser.add_argument("--num-test-tokens", type=int, default=NUM_TEST_TOKENS)
    parser.add_argument("--num-experts", type=int, default=SmolMoEConfig.num_experts)
    parser.add_argument(
        "--router-noise-std",
        type=float,
        default=SmolMoEConfig.router_noise_std,
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    set_seed(args.seed)
    device = get_device()

    # Upcycling and the parity check are done in float32: weight copying is
    # exact, and the check compares logits at full precision.
    print(f"Loading dense model {args.model_name} (float32)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dense_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32
    ).to(device).eval()

    config = SmolMoEConfig(
        num_experts=args.num_experts,
        router_noise_std=args.router_noise_std,
    )
    moe_model = SmolMoELM(config).to(device)
    print(
        f"Dense params: {count_parameters(dense_model):,} | "
        f"MoE params: {count_parameters(moe_model):,} "
        f"({config.num_experts} experts, top-{config.num_experts_per_tok})"
    )

    print("Upcycling weights...")
    upcycle_dense_to_moe(dense_model, moe_model)

    # --- Parity check ---------------------------------------------------------
    input_ids = tokenizer(args.test_prompt, return_tensors="pt").input_ids.to(device)
    max_err = check_upcycling(dense_model, moe_model, input_ids, args.parity_tolerance)
    print(f"Sanity check PASSED: max |Δlogit| = {max_err:.2e} < {args.parity_tolerance}")

    # --- Generation comparison (should be identical) ----------------------------
    moe_ids = moe_model.generate(
        input_ids, max_new_tokens=args.num_test_tokens, eos_token_id=tokenizer.eos_token_id
    )
    dense_ids = dense_model.generate(
        input_ids,
        max_new_tokens=args.num_test_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(f"\nPrompt: {args.test_prompt!r}")
    print(f"Dense : {tokenizer.decode(dense_ids[0], skip_special_tokens=True)!r}")
    print(f"MoE   : {tokenizer.decode(moe_ids[0], skip_special_tokens=True)!r}")

    save_moe_model(moe_model, args.output_path)  # weights bundled with config
    print(f"\nSaved upcycled MoE checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
