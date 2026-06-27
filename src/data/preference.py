"""Preference-pair construction for DPO/CPO.

Two generations are sampled per prompt at different temperatures; the variant
closer to the ground-truth correction (by character edit distance) becomes
``chosen`` and the other ``rejected``. Ties are dropped — they carry no
preference signal.
"""

import torch
from datasets import Dataset

from src.data.gec import extract_completion, format_prompt

try:
    from fast_edit_distance import edit_distance
except ImportError:  # pragma: no cover - pure-python fallback

    def edit_distance(a: str, b: str) -> int:
        """Levenshtein distance fallback if ``fast_edit_distance`` is absent."""
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = curr
        return prev[-1]


def generate_in_batches(
    model,
    tokenizer,
    sources: list[str],
    device: torch.device,
    generate_params: dict | None = None,
    initial_batch_size: int = 64,
    max_input_length: int = 128,
    verbose: bool = True,
) -> list[str]:
    """Generate a completion for every source, halving the batch size on OOM.

    Args:
        sources: raw GEC source strings (prompt formatting is applied here).
        generate_params: kwargs forwarded to ``model.generate``.
    """
    if generate_params is None:
        generate_params = {"max_new_tokens": 128, "do_sample": False}

    batch_size = initial_batch_size
    start = 0
    outputs: list[str] = []
    model.eval()

    while start < len(sources):
        end = min(start + batch_size, len(sources))
        batch = [format_prompt(text) for text in sources[start:end]]
        try:
            model_inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    **generate_params,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs.extend(extract_completion(text) for text in decoded)
            start = end
            if verbose and (start // max(batch_size, 1)) % 10 == 0:
                print(f"  generated {start}/{len(sources)}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise  # cannot shrink further; surface the real error
            batch_size = max(1, batch_size // 2)
            if verbose:
                print(f"  CUDA OOM — retrying with batch size {batch_size}")

    return outputs


def create_preference_dataset(
    prompts: list[str],
    targets: list[str],
    variants_1: list[str],
    variants_2: list[str],
) -> Dataset:
    """Build a chosen/rejected preference dataset from two generation variants.

    ``prompt`` is stored in inference format (with the separator) so the
    prompt seen during DPO/CPO training matches the prompt used at inference
    and evaluation time.
    """
    data = {"prompt": [], "chosen": [], "rejected": []}
    for prompt, target, v1, v2 in zip(prompts, targets, variants_1, variants_2):
        d1 = edit_distance(v1, target)
        d2 = edit_distance(v2, target)
        if d1 == d2:  # no preference signal
            continue
        chosen, rejected = (v1, v2) if d1 < d2 else (v2, v1)
        data["prompt"].append(format_prompt(prompt))
        data["chosen"].append(chosen)
        data["rejected"].append(rejected)
    return Dataset.from_dict(data)
