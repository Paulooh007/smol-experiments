"""07 — Analyze how routing and per-domain quality changed after specialization.

Loads the pre-specialization (05) and post-specialization (06) checkpoints
and compares them on the *held-out* validation split. The split is at the
document level (documents are split before chunking, so no document
contributes blocks to both sides), the data settings below match script 06
exactly, and the split is deterministic in those settings — so the
validation blocks here come from documents script 06 never trained on.

    1. Routing matrices at the middle decoder layer: for each domain
       (Code/Math/Chat), what percentage of its held-out tokens is routed to
       each expert — before vs after, printed as plain-text tables. After
       specialization the matrix should be close to diagonal
       (Code→Expert 0, Math→Expert 1, Chat→Expert 2).
    2. Per-domain held-out LM loss, before vs after — the functional check
       that specialization helped (routing % alone only shows the router
       obeyed its loss).
    3. Per-token routing on 9 example sentences (3 per domain), showing which
       expert each token of fresh, unseen text is sent to.

Analysis only — no training. Runs fine on CPU.

Prerequisites: run scripts/05 and scripts/06 first.
Run from the repo root:  uv run python scripts/07_analyze_routing.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from src.data.specialization import (
    DOMAIN_MAP,
    DOMAIN_NAMES,
    build_domain_val_loaders,
    build_specialization_dataloaders,
)
from src.checkpoint import load_moe_model
from src.evaluation import evaluate_per_domain_loss
from src.models import SmolMoELM
from src.utils import get_autocast_dtype, get_device, set_seed
from src.visualization import (
    routing_dashboard_html,
    write_domain_losses_csv,
    write_matrix_csv,
    write_token_routing_csv,
)

# ----------------------------- Configuration ----------------------------------
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # tokenizer only
BEFORE_PATH = REPO_ROOT / "outputs" / "moe_pretrained.pt"
AFTER_PATH = REPO_ROOT / "outputs" / "moe_specialized.pt"
ARTIFACTS_DIR = REPO_ROOT / "outputs" / "routing_analysis"

SEED = 42
# Data settings: MUST match script 06 so the deterministic document-level
# split reproduces the exact same held-out blocks.
BLOCK_SIZE = 256
MAX_SAMPLES_PER_DOMAIN = 500
VAL_FRACTION = 0.1
DATA_SEED = 100
BATCH_SIZE = 4

# Example sentences for per-token routing (3 per domain).
EXAMPLES = {
    "code": [
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "class Stack: def push(self, item): self.items.append(item)",
        "squares = [x**2 for x in range(10)]",
    ],
    "math": [
        "The integral of sin(x) from 0 to pi equals 2.",
        "Solve x^2 - 5x + 6 = 0 using the quadratic formula.",
        "If P(A) = 0.3 and P(B) = 0.5, find P(A and B) given P(A|B) = 0.3.",
    ],
    "chat": [
        "Can you recommend a good science fiction movie for the weekend?",
        "I'm planning a trip to Kenya, what should I pack?",
        "What's the difference between cold brew and iced coffee?",
    ],
}
# --------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--before-path", type=Path, default=BEFORE_PATH)
    parser.add_argument("--after-path", type=Path, default=AFTER_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--max-samples-per-domain", type=int, default=MAX_SAMPLES_PER_DOMAIN)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Decoder layer to inspect. Defaults to the middle layer.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Only print analysis; do not write CSV/HTML artifacts.",
    )
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> SmolMoELM:
    """Load a checkpoint produced by scripts 05/06 (config bundled inside)."""
    return load_moe_model(path).to(device).eval()


@torch.no_grad()
def compute_routing_matrix(
    model: SmolMoELM,
    loader,
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """[num_domains, num_experts] matrix: % of each domain's tokens per expert.

    Counts only real (non-padding) tokens, at one decoder layer, over the
    entire held-out loader.
    """
    num_experts = model.config.num_experts
    num_domains = len(DOMAIN_MAP)
    counts = torch.zeros(num_domains, num_experts)

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        selected = outputs["router_logits"][layer_idx].argmax(dim=-1).cpu()  # [B, T]
        domain_ids = batch["domain_ids"]  # [B, T]; pads are -100
        for d in range(num_domains):
            mask = domain_ids == d
            if mask.any():
                counts[d] += torch.bincount(
                    selected[mask], minlength=num_experts
                ).float()

    row_sums = counts.sum(dim=1, keepdim=True).clamp(min=1)
    return 100.0 * counts / row_sums


def print_routing_matrix(matrix: torch.Tensor, title: str) -> None:
    """Plain-text table: rows = domains, columns = experts."""
    num_experts = matrix.shape[1]
    print(f"\n{title}")
    header = "Domain".ljust(8) + "".join(
        f"Expert {e}".rjust(12) for e in range(num_experts)
    )
    print(header)
    print("-" * len(header))
    for d in range(matrix.shape[0]):
        row = DOMAIN_NAMES[d].ljust(8) + "".join(
            f"{matrix[d, e].item():11.1f}%" for e in range(num_experts)
        )
        print(row)


@torch.no_grad()
def collect_per_token_routing(
    model: SmolMoELM, tokenizer, layer_idx: int, device: torch.device
) -> list[dict]:
    """Collect token -> expert assignments for example sentences."""
    examples = []
    for domain, sentences in EXAMPLES.items():
        for sentence in sentences:
            enc = tokenizer(sentence, return_tensors="pt")
            outputs = model(input_ids=enc.input_ids.to(device))
            selected = outputs["router_logits"][layer_idx].argmax(dim=-1)[0].cpu()
            tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
            examples.append(
                {
                    "domain": domain,
                    "domain_id": DOMAIN_MAP[domain],
                    "sentence": sentence,
                    "tokens": [
                        {
                            "token": tok.replace(chr(0x120), " "),
                            "expert": int(expert),
                        }
                        for tok, expert in zip(tokens, selected)
                    ],
                }
            )
    return examples


def print_per_token_routing(examples: list[dict], num_experts: int) -> None:
    """Print token -> expert assignments for example sentences."""
    for domain in EXAMPLES:
        print(f"\n--- {domain.capitalize()} examples "
              f"(designated: Expert {DOMAIN_MAP[domain]}) ---")
        for example in [item for item in examples if item["domain"] == domain]:
            pairs = " ".join(
                f"{item['token'].replace(' ', '_')}→E{item['expert']}"
                for item in example["tokens"]
            )
            counts = torch.zeros(num_experts)
            for item in example["tokens"]:
                counts[item["expert"]] += 1
            dominant = int(counts.argmax())
            print(f"\n  {example['sentence']!r}")
            print(f"  {pairs}")
            print(
                f"  dominant: Expert {dominant} "
                f"({100.0 * counts[dominant] / counts.sum():.0f}% of tokens)"
            )


def write_analysis_artifacts(
    artifacts_dir: Path,
    before_matrix: torch.Tensor,
    after_matrix: torch.Tensor,
    losses_before: dict[str, float],
    losses_after: dict[str, float],
    token_examples: list[dict],
    layer_idx: int,
) -> None:
    """Persist routing analysis as CSV and self-contained HTML."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    row_labels = [DOMAIN_NAMES[i] for i in range(len(DOMAIN_MAP))]
    column_labels = [f"Expert {i}" for i in range(before_matrix.shape[1])]

    write_matrix_csv(
        artifacts_dir / "routing_before.csv",
        before_matrix,
        row_labels,
        column_labels,
    )
    write_matrix_csv(
        artifacts_dir / "routing_after.csv",
        after_matrix,
        row_labels,
        column_labels,
    )
    write_domain_losses_csv(artifacts_dir / "domain_losses.csv", losses_before, losses_after)
    write_token_routing_csv(artifacts_dir / "token_routing.csv", token_examples)
    dashboard = routing_dashboard_html(
        title="MoE routing analysis",
        before_matrix=before_matrix,
        after_matrix=after_matrix,
        row_labels=row_labels,
        column_labels=column_labels,
        before_losses=losses_before,
        after_losses=losses_after,
        token_examples=token_examples,
        layer_idx=layer_idx,
    )
    (artifacts_dir / "routing_dashboard.html").write_text(dashboard)
    print(f"\nSaved routing analysis artifacts to {artifacts_dir}")


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    set_seed(args.seed)
    device = get_device()
    amp_dtype = get_autocast_dtype()

    for path, script in ((args.before_path, "05"), (args.after_path, "06")):
        if not path.exists():
            sys.exit(f"Missing {path}. Run `uv run python scripts/{script}_*.py` first.")

    print("Loading checkpoints...")
    before = load_model(args.before_path, device)
    after = load_model(args.after_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Reproduce script 06's split; only the held-out part is used below.
    _, val_loader, val_ds = build_specialization_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_samples_per_domain=args.max_samples_per_domain,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
    )
    domain_val_loaders = build_domain_val_loaders(val_ds, batch_size=args.batch_size)
    print(f"Held-out val blocks: {len(val_ds)}")

    # Routers behave similarly across layers; the middle layer is a
    # representative probe without averaging away per-layer detail.
    layer_idx = args.layer_idx
    if layer_idx is None:
        layer_idx = before.config.num_hidden_layers // 2
    if not 0 <= layer_idx < before.config.num_hidden_layers:
        sys.exit(
            f"--layer-idx must be between 0 and {before.config.num_hidden_layers - 1}; "
            f"got {layer_idx}."
        )
    print(f"Analyzing routing at layer {layer_idx} "
          f"(of {before.config.num_hidden_layers})")

    # --- Routing matrices on held-out blocks -------------------------------------
    print("\n================ Routing matrices (held-out data) ================")
    before_matrix = compute_routing_matrix(before, val_loader, layer_idx, device)
    after_matrix = compute_routing_matrix(after, val_loader, layer_idx, device)
    print_routing_matrix(before_matrix, "BEFORE specialization (after script 05)")
    print_routing_matrix(after_matrix, "AFTER specialization (after script 06)")
    print("\nTarget after specialization: Code→Expert 0, Math→Expert 1, "
          "Chat→Expert 2 (diagonal-dominant matrix).")

    # --- Per-domain held-out loss: the functional check --------------------------
    print("\n================ Per-domain val loss ================")
    losses_before = evaluate_per_domain_loss(before, domain_val_loaders, device, amp_dtype)
    losses_after = evaluate_per_domain_loss(after, domain_val_loaders, device, amp_dtype)
    header = "Domain".ljust(8) + "Before".rjust(10) + "After".rjust(10) + "Change".rjust(10)
    print(header)
    print("-" * len(header))
    for name in DOMAIN_MAP:
        b, a = losses_before[name], losses_after[name]
        print(f"{name.capitalize():<8}{b:>10.4f}{a:>10.4f}{a - b:>+10.4f}")
    print("(negative change = specialization improved that domain)")

    # --- Per-token routing on example sentences -----------------------------------
    print("\n============ Per-token routing (specialized model) ============")
    token_examples = collect_per_token_routing(after, tokenizer, layer_idx, device)
    print_per_token_routing(token_examples, after.config.num_experts)

    if not args.no_artifacts:
        write_analysis_artifacts(
            artifacts_dir=args.artifacts_dir,
            before_matrix=before_matrix,
            after_matrix=after_matrix,
            losses_before=losses_before,
            losses_after=losses_after,
            token_examples=token_examples,
            layer_idx=layer_idx,
        )


if __name__ == "__main__":
    main()
