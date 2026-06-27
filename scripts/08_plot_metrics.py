"""Generate a self-contained HTML dashboard from experiment CSV metrics.

By default this scans ``outputs/*metrics*.csv`` and writes
``outputs/training_dashboard.html``. The script uses only the Python standard
library so it works anywhere the training scripts run.

Run from the repo root:

    uv run python scripts/08_plot_metrics.py
"""

from __future__ import annotations

import argparse
import html
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "training_dashboard.html"

import sys

sys.path.insert(0, str(REPO_ROOT))

from src.metrics import as_float, read_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="*",
        type=Path,
        help="CSV metric files. Defaults to outputs/*metrics*.csv.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="smol-experiments training dashboard")
    return parser.parse_args()


def find_default_metrics() -> list[Path]:
    return sorted((REPO_ROOT / "outputs").glob("*metrics*.csv"))


def series_key(row: dict[str, str]) -> str:
    parts = [row.get("phase", ""), row.get("domain", "")]
    label = " / ".join(part for part in parts if part)
    return label or "run"


def numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    excluded = {"step", "phase", "domain"}
    columns = rows[0].keys() if rows else []
    result = []
    for col in columns:
        if col in excluded:
            continue
        if sum(as_float(row.get(col)) is not None for row in rows) >= 1:
            result.append(col)
    return result


def collect_series(rows: list[dict[str, str]], metric: str) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        y = as_float(row.get(metric))
        if y is None:
            continue
        x = as_float(row.get("step"))
        if x is None:
            x = float(idx)
        grouped[series_key(row)].append((x, y))
    return {
        key: sorted(points)
        for key, points in grouped.items()
        if points
    }


def chart_svg(title: str, grouped: dict[str, list[tuple[float, float]]]) -> str:
    width, height = 900, 320
    left, right, top, bottom = 58, 24, 34, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_points = [point for points in grouped.values() for point in points]
    if not all_points:
        return ""

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    y_pad = (max_y - min_y) * 0.08
    min_y -= y_pad
    max_y += y_pad

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    colors = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
    ]
    lines = []
    legend = []
    for i, (label, points) in enumerate(grouped.items()):
        color = colors[i % len(colors)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        if len(points) == 1:
            x, y = points[0]
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}" />')
        else:
            lines.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" '
                'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
            )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        )

    grid = []
    for tick in range(5):
        frac = tick / 4
        y = top + frac * plot_h
        value = max_y - frac * (max_y - min_y)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" />')
        grid.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.3g}</text>')
    for tick in range(5):
        frac = tick / 4
        x = left + frac * plot_w
        value = min_x + frac * (max_x - min_x)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" />')
        grid.append(f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle">{value:.0f}</text>')

    return f"""
    <section class="chart">
      <h2>{html.escape(title)}</h2>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <g class="grid">{''.join(grid)}</g>
        <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" />
        <line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" />
        {''.join(lines)}
        <text class="x-label" x="{left + plot_w / 2:.1f}" y="{height - 4}" text-anchor="middle">step</text>
      </svg>
      <div class="legend">{''.join(legend)}</div>
    </section>
    """


def build_dashboard(metric_paths: list[Path], title: str) -> str:
    sections = []
    for path in metric_paths:
        rows = read_metric_rows(path)
        if not rows:
            continue
        charts = []
        for metric in numeric_columns(rows):
            grouped = collect_series(rows, metric)
            charts.append(chart_svg(f"{path.name}: {metric}", grouped))
        if charts:
            sections.append("\n".join(charts))

    body = "\n".join(sections) or "<p>No numeric metrics found.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1 {{
      margin: 0 0 24px;
      font-size: 28px;
      font-weight: 700;
    }}
    .chart {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 18px 18px 14px;
      margin: 0 0 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 650;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .grid line {{
      stroke: #e5e7eb;
      stroke-width: 1;
    }}
    .grid text {{
      fill: #6b7280;
      font-size: 11px;
    }}
    .axis {{
      stroke: #475569;
      stroke-width: 1.5;
    }}
    .x-label {{
      fill: #475569;
      font-size: 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      color: #334155;
      font-size: 13px;
      margin-top: 4px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend i {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    {body}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    metric_paths = args.metrics or find_default_metrics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_dashboard(metric_paths, args.title))
    print(f"Wrote dashboard to {args.output}")


if __name__ == "__main__":
    main()
