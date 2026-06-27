"""Evaluation utilities: GLEU/BLEU for the GEC task, loss/expert metrics for MoE."""

import csv
import math
from collections import Counter
from pathlib import Path

import evaluate as hf_evaluate
import torch

from src.data.gec import extract_source_sentence, prompt_token_length
from src.data.preference import generate_in_batches
from src.losses import causal_lm_loss

try:
    from fast_edit_distance import edit_distance
except ImportError:  # pragma: no cover - pure-python fallback
    from src.data.preference import edit_distance


LENGTH_BUCKETS = ("short", "medium", "long")


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_gleu(
    sources: list[str],
    references: list[str],
    candidates: list[str],
    max_n: int = 4,
) -> float:
    """GLEU for grammatical error correction (Napoles et al., 2015/2016).

    BLEU is a poor GEC metric: corrected sentences mostly copy the source, so
    a model that changes *nothing* scores high. GLEU fixes this by penalizing
    candidate n-grams that match the source but not the reference — i.e. the
    errors the system failed to correct. Per sentence and order n:

        numerator_n = sum_g min(C[g], R[g])
                      - sum_g max(0, min(C[g], S[g]) - min(C[g], R[g]))

    (floored at 0), where C/R/S are candidate/reference/source n-gram counts.
    Statistics are summed over the corpus, combined as a geometric mean over
    n = 1..max_n, and multiplied by BLEU's brevity penalty against the
    reference length. Single-reference variant; whitespace tokenization
    (CoEdIT text is plain, untokenized English — adequate at this scale).
    """
    if not (len(sources) == len(references) == len(candidates)):
        raise ValueError("sources, references and candidates must be equal length")

    numerators = [0] * max_n
    denominators = [0] * max_n
    cand_len_total, ref_len_total = 0, 0

    for src, ref, cand in zip(sources, references, candidates):
        src_tok, ref_tok, cand_tok = src.split(), ref.split(), cand.split()
        cand_len_total += len(cand_tok)
        ref_len_total += len(ref_tok)
        for n in range(1, max_n + 1):
            cand_counts = _ngram_counts(cand_tok, n)
            if not cand_counts:
                continue
            ref_counts = _ngram_counts(ref_tok, n)
            src_counts = _ngram_counts(src_tok, n)
            matches, penalties = 0, 0
            for gram, count in cand_counts.items():
                in_ref = min(count, ref_counts[gram])
                in_src = min(count, src_counts[gram])
                matches += in_ref
                penalties += max(0, in_src - in_ref)
            numerators[n - 1] += max(0, matches - penalties)
            denominators[n - 1] += sum(cand_counts.values())

    log_precision_sum = 0.0
    for num, den in zip(numerators, denominators):
        if den == 0:
            return 0.0
        # Standard smoothing for empty corpus-level counts (rare for n<=4).
        p_n = num / den if num > 0 else 1.0 / (2.0 * den)
        log_precision_sum += math.log(p_n)

    if cand_len_total == 0:
        return 0.0
    brevity_penalty = (
        1.0
        if cand_len_total >= ref_len_total
        else math.exp(1.0 - ref_len_total / cand_len_total)
    )
    return brevity_penalty * math.exp(log_precision_sum / max_n)


def gec_length_bucket(token_count: int) -> str:
    """Bucket formatted prompt lengths for small-model GEC evaluation."""
    if token_count <= 128:
        return "short"
    if token_count <= 256:
        return "medium"
    return "long"


def _summarize_gec_predictions(
    sources: list[str],
    references: list[str],
    predictions: list[str],
    bleu_metric,
    edit_to_ref: list[int] | None = None,
    edit_to_source: list[int] | None = None,
) -> dict[str, float | int]:
    n = len(predictions)
    if n == 0:
        return {
            "n": 0,
            "gleu": 0.0,
            "bleu": 0.0,
            "exact_copy_rate": 0.0,
            "changed_rate": 0.0,
            "empty_rate": 0.0,
            "avg_edit_distance_to_reference": 0.0,
            "avg_edit_distance_to_source": 0.0,
        }

    if edit_to_ref is None:
        edit_to_ref = [
            edit_distance(pred, ref) for pred, ref in zip(predictions, references)
        ]
    if edit_to_source is None:
        edit_to_source = [
            edit_distance(pred, source) for pred, source in zip(predictions, sources)
        ]

    empty_count = sum(1 for pred in predictions if not pred.strip())
    copy_count = sum(
        1 for source, pred in zip(sources, predictions) if pred.strip() == source.strip()
    )
    bleu = bleu_metric.compute(predictions=predictions, references=references)
    return {
        "n": n,
        "gleu": corpus_gleu(sources, references, predictions),
        "bleu": bleu["bleu"],
        "exact_copy_rate": copy_count / n,
        "changed_rate": 1.0 - (copy_count / n),
        "empty_rate": empty_count / n,
        "avg_edit_distance_to_reference": sum(edit_to_ref) / n,
        "avg_edit_distance_to_source": sum(edit_to_source) / n,
    }


def evaluate_gec(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    generate_params: dict | None = None,
    initial_batch_size: int = 64,
    max_input_length: int = 512,
    output_path: Path | None = None,
) -> dict:
    """Batch-generate corrections for ``dataset['src']`` and score GLEU + BLEU.

    GLEU is the headline metric (it penalizes uncorrected errors that BLEU
    rewards — see :func:`corpus_gleu`); BLEU is reported alongside for
    comparability with the original notebooks. Greedy decoding by default —
    evaluation should be deterministic so SFT, DPO and CPO numbers are
    comparable (the notebook sampled with ``do_sample=True``, adding noise).
    """
    if generate_params is None:
        generate_params = {"max_new_tokens": 128, "do_sample": False}
    raw_sources = list(dataset["src"])
    preds = generate_in_batches(
        model,
        tokenizer,
        raw_sources,
        device=device,
        generate_params=generate_params,
        initial_batch_size=initial_batch_size,
        max_input_length=max_input_length,
    )
    references = list(dataset["tgt"])
    sources = [extract_source_sentence(s) for s in raw_sources]
    edit_to_ref = [edit_distance(pred, ref) for pred, ref in zip(preds, references)]
    edit_to_source = [edit_distance(pred, source) for pred, source in zip(preds, sources)]
    prompt_lengths: list[int] = []
    length_buckets: list[str] = []
    if callable(tokenizer):
        prompt_lengths = [prompt_token_length(src, tokenizer) for src in raw_sources]
        length_buckets = [gec_length_bucket(length) for length in prompt_lengths]

    bleu_metric = hf_evaluate.load("bleu")
    if output_path is not None:
        write_gec_predictions(
            output_path=output_path,
            raw_sources=raw_sources,
            sources=sources,
            references=references,
            predictions=preds,
            edit_to_ref=edit_to_ref,
            edit_to_source=edit_to_source,
            prompt_token_lengths=prompt_lengths or None,
            length_buckets=length_buckets or None,
        )

    scores = _summarize_gec_predictions(
        sources=sources,
        references=references,
        predictions=preds,
        bleu_metric=bleu_metric,
        edit_to_ref=edit_to_ref,
        edit_to_source=edit_to_source,
    )
    if length_buckets:
        scores["length_buckets"] = {}
        for bucket_name in LENGTH_BUCKETS:
            idxs = [i for i, bucket in enumerate(length_buckets) if bucket == bucket_name]
            scores["length_buckets"][bucket_name] = _summarize_gec_predictions(
                sources=[sources[i] for i in idxs],
                references=[references[i] for i in idxs],
                predictions=[preds[i] for i in idxs],
                bleu_metric=bleu_metric,
                edit_to_ref=[edit_to_ref[i] for i in idxs],
                edit_to_source=[edit_to_source[i] for i in idxs],
            )
    return scores


def write_gec_predictions(
    output_path: Path,
    raw_sources: list[str],
    sources: list[str],
    references: list[str],
    predictions: list[str],
    edit_to_ref: list[int],
    edit_to_source: list[int],
    prompt_token_lengths: list[int] | None = None,
    length_buckets: list[str] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if prompt_token_lengths is None:
        prompt_token_lengths = [None] * len(predictions)
    if length_buckets is None:
        length_buckets = [""] * len(predictions)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "raw_src",
                "source_sentence",
                "reference",
                "prediction",
                "prompt_token_length",
                "length_bucket",
                "copied_source",
                "empty_prediction",
                "edit_distance_to_reference",
                "edit_distance_to_source",
            ],
        )
        writer.writeheader()
        for raw, source, ref, pred, token_len, bucket, dist_ref, dist_source in zip(
            raw_sources,
            sources,
            references,
            predictions,
            prompt_token_lengths,
            length_buckets,
            edit_to_ref,
            edit_to_source,
        ):
            writer.writerow(
                {
                    "raw_src": raw,
                    "source_sentence": source,
                    "reference": ref,
                    "prediction": pred,
                    "prompt_token_length": "" if token_len is None else token_len,
                    "length_bucket": bucket,
                    "copied_source": pred.strip() == source.strip(),
                    "empty_prediction": not pred.strip(),
                    "edit_distance_to_reference": dist_ref,
                    "edit_distance_to_source": dist_source,
                }
            )


@torch.no_grad()
def evaluate_moe_loss(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> float:
    """Average causal LM loss of a custom model over a DataLoader."""
    was_training = model.training
    model.eval()
    total, batches = 0.0, 0
    use_amp = device.type == "cuda" and amp_dtype is not None
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = causal_lm_loss(outputs["logits"], input_ids, attention_mask)
        total += loss.item()
        batches += 1
    if was_training:
        model.train()
    return total / max(batches, 1)


def compute_active_expert_pct(all_router_logits: list[torch.Tensor]) -> float:
    """Percentage of (layer, expert) slots that received at least one token.

    100% means every expert in every layer is active; lower values indicate
    dead experts — the signature of router collapse.
    """
    active, total = 0, 0
    with torch.no_grad():
        for router_logits in all_router_logits:
            num_experts = router_logits.shape[-1]
            counts = torch.bincount(
                router_logits.argmax(dim=-1).reshape(-1), minlength=num_experts
            )
            active += int((counts > 0).sum())
            total += num_experts
    return 100.0 * active / max(total, 1)


def evaluate_per_domain_loss(
    model,
    domain_loaders: dict,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Average causal LM loss on each domain's held-out blocks.

    This is the functional test of specialization: routing matrices show that
    the router *obeyed* the specialization loss, while per-domain loss shows
    whether the specialized model actually got better at each domain.
    """
    return {
        name: evaluate_moe_loss(model, loader, device, amp_dtype)
        for name, loader in domain_loaders.items()
    }
