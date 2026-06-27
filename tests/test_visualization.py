from pathlib import Path

from src.metrics import read_metric_rows
from src.visualization import (
    expert_label,
    routing_dashboard_html,
    viridis_color,
    write_domain_losses_csv,
    write_matrix_csv,
    write_token_routing_csv,
)


def test_expert_label_includes_default_domain_mapping():
    assert expert_label(0) == "Expert 0 (Code)"
    assert expert_label(1) == "Expert 1 (Math)"
    assert expert_label(9) == "Expert 9"


def test_write_matrix_csv_with_labels(tmp_path: Path):
    path = tmp_path / "routing.csv"

    write_matrix_csv(
        path,
        [[75.0, 25.0], [10.0, 90.0]],
        row_labels=["Code", "Math"],
        column_labels=["Expert 0", "Expert 1"],
    )

    rows = read_metric_rows(path)
    assert rows[0]["label"] == "Code"
    assert rows[0]["Expert 0"] == "75"
    assert rows[1]["Expert 1"] == "90"


def test_write_domain_losses_csv(tmp_path: Path):
    path = tmp_path / "domain_losses.csv"

    write_domain_losses_csv(path, {"code": 2.0}, {"code": 1.5})

    rows = read_metric_rows(path)
    assert rows == [
        {"domain": "code", "before": "2", "after": "1.5", "change": "-0.5"}
    ]


def test_write_token_routing_csv(tmp_path: Path):
    path = tmp_path / "token_routing.csv"
    examples = [
        {
            "domain": "code",
            "sentence": "x = 1",
            "tokens": [{"token": "x", "expert": 0}, {"token": "=", "expert": 1}],
        }
    ]

    write_token_routing_csv(path, examples)

    rows = read_metric_rows(path)
    assert rows[0]["token"] == "x"
    assert rows[1]["expert"] == "1"


def test_routing_dashboard_contains_heatmaps_and_tokens():
    html = routing_dashboard_html(
        title="Routing",
        before_matrix=[[50, 50], [20, 80]],
        after_matrix=[[90, 10], [5, 95]],
        row_labels=["Code", "Math"],
        column_labels=["Expert 0", "Expert 1"],
        before_losses={"code": 2.0},
        after_losses={"code": 1.5},
        token_examples=[
            {
                "domain": "code",
                "domain_id": 0,
                "sentence": "x = 1",
                "tokens": [{"token": "x", "expert": 0}],
            }
        ],
        layer_idx=4,
    )

    assert "Routing" in html
    assert "90.0%" in html
    assert "Expert 0 (Code)" in html
    assert "Target: Expert 0 (Code)" in html
    assert "Per-token routing examples" in html
    assert "x = 1" in html
    assert "decoder layer 4" in html


def test_viridis_color_returns_hex_color():
    color = viridis_color(0.5, 0.0, 1.0)

    assert color.startswith("#")
    assert len(color) == 7
