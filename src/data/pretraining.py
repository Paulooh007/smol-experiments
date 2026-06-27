"""Continued pre-training data: Cosmopedia-100k tokenized into fixed blocks.

Documents are tokenized (with a trailing EOS), chunked per-document into
non-overlapping ``block_size`` windows, and the trailing remainder of each
document is dropped. Every resulting block is fully packed, so its attention
mask is all ones.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def build_pretraining_datasets(
    tokenizer,
    dataset_id: str = "HuggingFaceTB/cosmopedia-100k",
    split: str = "train",
    text_column: str = "text",
    block_size: int = 256,
    max_samples: int = 1000,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    """Tokenize and chunk a text dataset; return (train_ds, val_ds)."""
    ds = load_dataset(dataset_id, split=split)
    ds = ds.select(range(min(max_samples, len(ds))))
    eos = tokenizer.eos_token_id

    def tokenize(batch):
        out = tokenizer(batch[text_column], add_special_tokens=False)
        return {"input_ids": [ids + [eos] for ids in out["input_ids"]]}

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    def chunk_per_doc(batch):
        out_ids = []
        for ids in batch["input_ids"]:
            usable = (len(ids) // block_size) * block_size
            for i in range(0, usable, block_size):
                out_ids.append(ids[i : i + block_size])
        return {
            "input_ids": out_ids,
            "attention_mask": [[1] * block_size for _ in out_ids],
        }

    ds = ds.map(chunk_per_doc, batched=True)

    split_ds = ds.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
    train_ds, val_ds = split_ds["train"], split_ds["test"]
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return train_ds, val_ds


def build_pretraining_dataloaders(
    tokenizer,
    batch_size: int = 4,
    device: torch.device | None = None,
    **dataset_kwargs,
):
    """Convenience wrapper returning (train_loader, val_loader)."""
    train_ds, val_ds = build_pretraining_datasets(tokenizer, **dataset_kwargs)
    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin
    )
    return train_loader, val_loader
