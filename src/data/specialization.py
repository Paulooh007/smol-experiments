"""Domain-labelled data for expert specialization, split at the document level.

Code/Math/Chat examples are streamed from the Llama-Nemotron post-training
dataset and tokenized into documents. Within each domain, *documents* are
deterministically split into train/validation, and only then chunked into
fixed-size blocks tagged with per-token ``domain_id``s. Splitting before
chunking matters: adjacent blocks of the same document share style,
vocabulary and boilerplate, so a block-level split leaks information from
training documents into "held-out" blocks. With a document-level split, no
document contributes blocks to both sides.

The split is per-domain (each domain's documents are split independently),
which guarantees every domain is represented in the validation set
regardless of how document counts differ between domains.

The materialized train set is shuffled by the DataLoader, so every batch
contains a mix of domains — the router gets contrastive signal from all
domains in (almost) every gradient step.

Padding convention: short documents are right-padded into a single block;
pad positions get ``attention_mask = 0`` and ``domain_id = -100`` so they are
excluded from both the LM loss and the router specialization loss. Each
block comes from a single document, so all real tokens in a block share one
domain.

Reproducibility contract: ``build_specialization_datasets`` is deterministic
in (tokenizer, block_size, max_samples_per_domain, val_fraction, seed).
Scripts 06 and 07 call it with identical arguments, which reproduces the
exact same split — so script 07's "validation" blocks are guaranteed to come
from documents script 06 never trained on.
"""

import random

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

DOMAIN_MAP = {"code": 0, "math": 1, "chat": 2}
DOMAIN_NAMES = {v: k.capitalize() for k, v in DOMAIN_MAP.items()}
PAD_DOMAIN_ID = -100  # ignored by router_specialization_loss


def _extract_text(example: dict) -> str:
    """Nemotron 'input' is a list of chat turns; join their contents."""
    input_data = example["input"]
    if isinstance(input_data, list):
        parts = [turn.get("content", "") for turn in input_data if isinstance(turn, dict)]
        return "\n".join(parts)
    return str(input_data)


def domain_documents(tokenizer, domain_name, max_samples, seed=100):
    """Yield tokenized documents (lists of token ids, EOS-terminated) for one domain."""
    eos = tokenizer.eos_token_id
    stream = (
        load_dataset(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "SFT",
            split=domain_name,
            streaming=True,
        )
        .shuffle(buffer_size=10_000, seed=seed)
        .take(max_samples)
    )
    for example in stream:
        token_ids = tokenizer(_extract_text(example), add_special_tokens=False)["input_ids"]
        yield token_ids + [eos]


def chunk_document(token_ids: list[int], domain_id: int, block_size: int, pad_id: int) -> list[dict]:
    """Chunk one document into fixed-size blocks with per-token domain labels.

    Documents shorter than ``block_size`` are right-padded into a single block
    (instead of being dropped, so no domain is starved of data); pads are
    masked out of all losses. For longer documents, the tail remainder is
    dropped.
    """
    length = len(token_ids)
    if length < block_size:
        n_pad = block_size - length
        return [{
            "input_ids": token_ids + [pad_id] * n_pad,
            "attention_mask": [1] * length + [0] * n_pad,
            "domain_ids": [domain_id] * length + [PAD_DOMAIN_ID] * n_pad,
        }]
    usable = (length // block_size) * block_size
    return [
        {
            "input_ids": token_ids[i : i + block_size],
            "attention_mask": [1] * block_size,
            "domain_ids": [domain_id] * block_size,
        }
        for i in range(0, usable, block_size)
    ]


def build_specialization_datasets(
    tokenizer,
    block_size: int = 256,
    max_samples_per_domain: int = 500,
    val_fraction: float = 0.1,
    seed: int = 100,
) -> tuple[Dataset, Dataset]:
    """Materialize domain blocks into (train_ds, val_ds), split by document.

    For each domain: stream documents, shuffle them with a seeded RNG, hold
    out ``val_fraction`` of *documents* for validation, then chunk each side
    into blocks. A few thousand blocks of ``block_size`` ints — small enough
    to hold in memory.
    """
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    train_blocks: list[dict] = []
    val_blocks: list[dict] = []

    for name, did in DOMAIN_MAP.items():
        docs = list(domain_documents(tokenizer, name, max_samples_per_domain, seed))
        rng = random.Random(seed + did)  # per-domain, deterministic
        rng.shuffle(docs)
        n_val = max(1, round(len(docs) * val_fraction))
        for doc in docs[:n_val]:
            val_blocks.extend(chunk_document(doc, did, block_size, pad_id))
        for doc in docs[n_val:]:
            train_blocks.extend(chunk_document(doc, did, block_size, pad_id))

    train_ds = Dataset.from_list(train_blocks)
    val_ds = Dataset.from_list(val_blocks)
    for d in (train_ds, val_ds):
        d.set_format(type="torch", columns=["input_ids", "attention_mask", "domain_ids"])
    return train_ds, val_ds


def build_specialization_dataloaders(
    tokenizer,
    batch_size: int = 4,
    block_size: int = 256,
    max_samples_per_domain: int = 500,
    val_fraction: float = 0.1,
    seed: int = 100,
) -> tuple[DataLoader, DataLoader, Dataset]:
    """Returns (train_loader, val_loader, val_ds).

    ``val_ds`` is returned alongside the loaders so callers can derive
    per-domain views of the same held-out blocks (see
    ``build_domain_val_loaders``).
    """
    train_ds, val_ds = build_specialization_datasets(
        tokenizer,
        block_size=block_size,
        max_samples_per_domain=max_samples_per_domain,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_ds


def build_domain_val_loaders(val_ds: Dataset, batch_size: int = 4) -> dict[str, DataLoader]:
    """Split the held-out set into one DataLoader per domain.

    Blocks are domain-pure (each comes from a single document) and padding is
    on the right, so a block's domain is its first token's ``domain_id``.
    """
    loaders: dict[str, DataLoader] = {}
    for name, did in DOMAIN_MAP.items():
        domain_ds = val_ds.filter(lambda ex, did=did: int(ex["domain_ids"][0]) == did)
        loaders[name] = DataLoader(domain_ds, batch_size=batch_size, shuffle=False)
    return loaders
