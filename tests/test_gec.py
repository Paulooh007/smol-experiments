from pathlib import Path

from datasets import Dataset

from src.data.gec import (
    extract_completion,
    extract_source_sentence,
    filter_by_prompt_token_length,
    filter_by_sft_token_length,
    format_for_sft_completion,
    format_prompt,
)
import src.evaluation as evaluation
from src.evaluation import gec_length_bucket, write_gec_predictions
from src.metrics import read_metric_rows


class WhitespaceTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": text.split()}


def test_format_prompt_uses_explicit_correction_prefix():
    prompt = format_prompt("Fix grammatical errors in this sentence: He go home.")

    assert prompt == "Fix grammatical errors in this sentence: He go home.\nCorrection:"


def test_format_for_sft_completion_adds_leading_space_and_eos():
    row = format_for_sft_completion(
        {"src": "Fix grammatical errors in this sentence: He go home.", "tgt": "He goes home."},
        eos_token="<eos>",
    )

    assert row["prompt"].endswith("\nCorrection:")
    assert row["completion"] == " He goes home.<eos>"


def test_extract_completion_from_decoded_prompt_plus_answer():
    decoded = "Fix grammatical errors in this sentence: He go home.\nCorrection: He goes home."

    assert extract_completion(decoded) == "He goes home."


def test_extract_source_sentence_removes_coedit_instruction():
    assert (
        extract_source_sentence("Fix grammatical errors in this sentence: He go home.")
        == "He go home."
    )


def test_token_length_filters_keep_short_examples():
    dataset = Dataset.from_dict(
        {
            "src": [
                "Fix grammatical errors in this sentence: He go.",
                "Fix grammatical errors in this sentence: " + "word " * 20,
            ],
            "tgt": ["He goes.", "word " * 20],
        }
    )
    tokenizer = WhitespaceTokenizer()

    prompt_filtered = filter_by_prompt_token_length(dataset, tokenizer, max_tokens=9)
    sft_filtered = filter_by_sft_token_length(
        dataset,
        tokenizer,
        max_tokens=11,
        eos_token=tokenizer.eos_token,
    )

    assert len(prompt_filtered) == 1
    assert len(sft_filtered) == 1


def test_gec_length_bucket_boundaries():
    assert gec_length_bucket(128) == "short"
    assert gec_length_bucket(129) == "medium"
    assert gec_length_bucket(256) == "medium"
    assert gec_length_bucket(257) == "long"


def test_write_gec_predictions_outputs_inspection_csv(tmp_path: Path):
    output = tmp_path / "predictions.csv"

    write_gec_predictions(
        output_path=output,
        raw_sources=["Fix grammatical errors in this sentence: He go home."],
        sources=["He go home."],
        references=["He goes home."],
        predictions=["He goes home."],
        edit_to_ref=[0],
        edit_to_source=[2],
    )

    rows = read_metric_rows(output)
    assert rows[0]["prediction"] == "He goes home."
    assert rows[0]["copied_source"] == "False"
    assert rows[0]["edit_distance_to_reference"] == "0"


def test_evaluate_gec_forwards_max_input_length(monkeypatch):
    captured = {}

    def fake_generate_in_batches(*args, **kwargs):
        captured["max_input_length"] = kwargs["max_input_length"]
        return ["He goes home."]

    class FakeBleu:
        def compute(self, predictions, references):
            return {"bleu": 1.0}

    monkeypatch.setattr(evaluation, "generate_in_batches", fake_generate_in_batches)
    monkeypatch.setattr(evaluation.hf_evaluate, "load", lambda name: FakeBleu())
    dataset = {
        "src": ["Fix grammatical errors in this sentence: He go home."],
        "tgt": ["He goes home."],
    }

    scores = evaluation.evaluate_gec(
        model=object(),
        tokenizer=WhitespaceTokenizer(),
        dataset=dataset,
        device=object(),
        max_input_length=777,
    )

    assert captured["max_input_length"] == 777
    assert scores["empty_rate"] == 0.0
    assert scores["length_buckets"]["short"]["n"] == 1
