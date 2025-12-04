import torch
from datasets import load_dataset, interleave_datasets, IterableDataset as HF_IterableDataset, Dataset
from fast_edit_distance import edit_distance

def format_text(example):
    """Applies chat template for SFT: 'src ###> tgt'"""
    if isinstance(example, str):
        return example + " ###>"
    
    messages = f"{example['src']} ###>{example['tgt']}"
    example["text"] = messages
    return example

def create_dpo_dataset(prompts, targets, variants1, variants2):
    """
    Creates a preference dataset by comparing edit distances of two model outputs
    against the ground truth.
    """
    data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "target": [],
    }

    for prompt, target, var1, var2 in zip(prompts, targets, variants1, variants2):
        dist1 = edit_distance(var1, target)
        dist2 = edit_distance(var2, target)

        if dist1 == dist2:
            continue

        if dist1 < dist2:
            chosen, rejected = var1, var2
        else:
            chosen, rejected = var2, var1

        data["prompt"].append(prompt)
        data["chosen"].append(chosen)
        data["rejected"].append(rejected)
        data["target"].append(target)

    return Dataset.from_dict(data)

def build_cosmopedia_dataset(tokenizer, block_size=256, max_samples=1000):
    """
    Builds the dataset for continuous pre-training (Cosmopedia).
    """
    ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
    ds = ds.select(range(max_samples))
    
    EOS = tokenizer.eos_token_id

    def tok(batch):
        out = tokenizer(batch["text"], add_special_tokens=False, return_attention_mask=True)
        out["input_ids"] = [ids + [EOS] for ids in out["input_ids"]]
        out["attention_mask"] = [m + [1] for m in out["attention_mask"]]
        return out

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

    def group_per_doc(batch):
        out_ids = []
        for ids in batch["input_ids"]:
            L = len(ids)
            n = (L // block_size) * block_size
            for i in range(0, n, block_size):
                out_ids.append(ids[i:i+block_size])
        return {"input_ids": out_ids, "attention_mask": [[1]*len(o) for o in out_ids]}

    ds = ds.map(group_per_doc, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds

# --- Expert Specialization Helpers ---

def domain_generator(tokenizer, block_size, domain_name, domain_id, max_samples):
    """
    Generator for streaming domain-specific data with domain IDs.
    """
    EOS = tokenizer.eos_token_id
    PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else EOS

    ds_stream = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "SFT",
        split=domain_name,
        streaming=True
    ).shuffle(buffer_size=10_000, seed=42).take(max_samples)

    for example in ds_stream:
        input_data = example["input"]
        if isinstance(input_data, list):
            content_parts = [turn.get('content', '') for turn in input_data if isinstance(turn, dict)]
            input_text = "\n".join(content_parts)
        else:
            input_text = str(input_data)

        token_ids = tokenizer(input_text, add_special_tokens=False)['input_ids'] + [EOS]
        
        L = len(token_ids)
        if L < block_size:
            padded_ids = token_ids + [PAD] * (block_size - L)
            attention_mask = [1] * L + [0] * (block_size - L)
            yield {
                "input_ids": padded_ids,
                "attention_mask": attention_mask,
                "domain_ids": [domain_id] * block_size
            }
        else:
            n = (L // block_size) * block_size
            for i in range(0, n, block_size):
                yield {
                    "input_ids": token_ids[i:i+block_size],
                    "attention_mask": [1] * block_size,
                    "domain_ids": [domain_id] * block_size
                }

def get_specialized_dataset(tokenizer, block_size=256, max_samples=500):
    """
    Returns an interleaved iterable dataset of Code, Math, and Chat.
    """
    domain_map = {"code": 0, "math": 1, "chat": 2}
    domain_datasets = [
        HF_IterableDataset.from_generator(
            domain_generator,
            gen_kwargs={
                "tokenizer": tokenizer,
                "block_size": block_size,
                "domain_name": name,
                "domain_id": did,
                "max_samples": max_samples
            }
        ) for name, did in domain_map.items()
    ]
    
    return interleave_datasets(
        domain_datasets,
        probabilities=[1/3, 1/3, 1/3],
        stopping_strategy="all_exhausted"
    )

def specialized_collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    domain_ids = torch.tensor([item['domain_ids'] for item in batch], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'domain_ids': domain_ids
    }