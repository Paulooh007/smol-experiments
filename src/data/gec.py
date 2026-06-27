"""Grammatical error correction data from the Grammarly CoEdIT dataset."""

from datasets import load_dataset

# Base SmolLM-135M has no chat template, so we use a plain prompt/completion
# format that is explicit and easy to reuse for SFT, generation, DPO and eval.
COMPLETION_PREFIX = "\nCorrection:"


def load_gec_datasets(max_train_samples: int | None = None):
    """Load CoEdIT and keep only the GEC task.

    Returns:
        (train_ds, test_ds) — ~19823 train and ~485 validation examples, each
        with ``src`` (instruction + errorful sentence) and ``tgt`` (correction).
    """
    train_ds = load_dataset("grammarly/coedit", split="train")
    test_ds = load_dataset("grammarly/coedit", split="validation")
    train_ds = train_ds.filter(lambda ex: ex["task"] == "gec")
    test_ds = test_ds.filter(lambda ex: ex["task"] == "gec")
    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    return train_ds, test_ds


def prompt_token_length(src: str, tokenizer) -> int:
    """Number of tokens in the inference prompt for a GEC source string."""
    return len(tokenizer(format_prompt(src), add_special_tokens=False)["input_ids"])


def sft_example_token_length(example: dict, tokenizer, eos_token: str) -> int:
    """Number of tokens in the SFT prompt plus completion."""
    row = format_for_sft_completion(example, eos_token)
    return len(tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"])


def filter_by_prompt_token_length(dataset, tokenizer, max_tokens: int | None):
    """Keep examples whose formatted inference prompt fits within ``max_tokens``."""
    if max_tokens is None:
        return dataset
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return dataset.filter(
        lambda ex: prompt_token_length(ex["src"], tokenizer) <= max_tokens,
        desc=f"Filtering examples with prompt <= {max_tokens} tokens",
    )


def filter_by_sft_token_length(dataset, tokenizer, max_tokens: int | None, eos_token: str):
    """Keep examples whose SFT prompt+completion fits within ``max_tokens``."""
    if max_tokens is None:
        return dataset
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return dataset.filter(
        lambda ex: sft_example_token_length(ex, tokenizer, eos_token) <= max_tokens,
        desc=f"Filtering SFT examples with prompt+completion <= {max_tokens} tokens",
    )


def format_for_sft_completion(example: dict, eos_token: str) -> dict:
    """Build prompt/completion fields for completion-only SFT.

    The leading space in the completion makes the answer read naturally after
    ``Correction:`` and usually tokenizes better than a bare first word.
    """
    return {
        "prompt": format_prompt(example["src"]),
        "completion": f" {example['tgt']}{eos_token}",
    }


def format_prompt(src: str) -> str:
    """Build the inference prompt for a source sentence."""
    return f"{src}{COMPLETION_PREFIX}"


def extract_completion(decoded_text: str) -> str:
    """Recover the model's correction from a decoded generation."""
    return decoded_text.split(COMPLETION_PREFIX.strip())[-1].strip()


def extract_source_sentence(src: str) -> str:
    """Strip the task instruction from a CoEdIT ``src`` field.

    CoEdIT sources look like ``"Fix grammatical errors in this sentence: He
    go to school."`` — instruction, colon, then the errorful sentence. GLEU
    needs the bare errorful sentence (the metric compares candidate n-grams
    against both the source and the reference), so the instruction must not
    count as "source text". Falls back to the full string if no colon is
    found.
    """
    _, sep, sentence = src.partition(": ")
    return sentence.strip() if sep else src.strip()
