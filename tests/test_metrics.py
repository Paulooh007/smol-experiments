from pathlib import Path
import importlib.util

from src.metrics import CsvMetricLogger, as_float, read_metric_rows


def load_plot_metrics_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "08_plot_metrics.py"
    spec = importlib.util.spec_from_file_location("plot_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_csv_metric_logger_writes_rows(tmp_path: Path):
    path = tmp_path / "metrics.csv"

    with CsvMetricLogger(path, ["phase", "step", "loss"]) as logger:
        logger.log(phase="train", step=1, loss=2.5)

    rows = read_metric_rows(path)
    assert rows == [{"phase": "train", "step": "1", "loss": "2.5"}]


def test_as_float_handles_blanks_and_numbers():
    assert as_float("") is None
    assert as_float("nope") is None
    assert as_float("3.5") == 3.5


def test_build_dashboard_renders_metric_chart(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    with CsvMetricLogger(path, ["phase", "step", "loss"]) as logger:
        logger.log(phase="train", step=1, loss=2.5)
        logger.log(phase="train", step=2, loss=2.0)

    plot_metrics = load_plot_metrics_module()
    html = plot_metrics.build_dashboard([path], "Test dashboard")

    assert "Test dashboard" in html
    assert "metrics.csv: loss" in html
