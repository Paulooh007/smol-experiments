"""Dependency-light visualization helpers for experiment artifacts.

The training scripts often run on Modal, where notebook display hooks and
plotting libraries may not be installed. These helpers emit plain CSV and
self-contained HTML so artifacts are easy to download and inspect anywhere.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Sequence

VIRIDIS_STOPS = [
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
]

EXPERT_COLORS = [
    "#b8860b",
    "#1e90ff",
    "#228b22",
    "#9333ea",
    "#dc2626",
    "#0891b2",
]

DEFAULT_EXPERT_DOMAINS = {
    0: "Code",
    1: "Math",
    2: "Chat",
}


def as_nested_list(matrix) -> list[list[float]]:
    """Convert tensors, numpy arrays, or nested sequences into float rows."""
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu()
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    return [[float(value) for value in row] for row in matrix]


def expert_label(expert_idx: int, expert_domains: dict[int, str] | None = None) -> str:
    """Human-readable label for an expert and its intended domain."""
    domain = (expert_domains or DEFAULT_EXPERT_DOMAINS).get(expert_idx)
    if domain:
        return f"Expert {expert_idx} ({domain})"
    return f"Expert {expert_idx}"


def write_matrix_csv(
    path: Path,
    matrix,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
) -> None:
    """Write a labelled matrix as CSV."""
    rows = as_nested_list(matrix)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *column_labels])
        for label, values in zip(row_labels, rows):
            writer.writerow([label, *[f"{value:.6g}" for value in values]])


def write_domain_losses_csv(
    path: Path,
    before: dict[str, float],
    after: dict[str, float],
) -> None:
    """Write before/after per-domain losses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "before", "after", "change"])
        writer.writeheader()
        for domain in before:
            writer.writerow(
                {
                    "domain": domain,
                    "before": f"{before[domain]:.6g}",
                    "after": f"{after[domain]:.6g}",
                    "change": f"{after[domain] - before[domain]:+.6g}",
                }
            )


def write_token_routing_csv(path: Path, examples: list[dict]) -> None:
    """Write per-token routing assignments as a flat CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain", "sentence", "token_index", "token", "expert"],
        )
        writer.writeheader()
        for example in examples:
            for i, item in enumerate(example["tokens"]):
                writer.writerow(
                    {
                        "domain": example["domain"],
                        "sentence": example["sentence"],
                        "token_index": i,
                        "token": item["token"],
                        "expert": item["expert"],
                    }
                )


def _lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def viridis_color(value: float, min_value: float, max_value: float) -> str:
    """Map a value onto a compact viridis-like color ramp."""
    if max_value <= min_value:
        t = 1.0
    else:
        t = min(1.0, max(0.0, (value - min_value) / (max_value - min_value)))
    scaled = t * (len(VIRIDIS_STOPS) - 1)
    idx = min(int(scaled), len(VIRIDIS_STOPS) - 2)
    local_t = scaled - idx
    c0, c1 = VIRIDIS_STOPS[idx], VIRIDIS_STOPS[idx + 1]
    rgb = tuple(_lerp(a, b, local_t) for a, b in zip(c0, c1))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def heatmap_table(
    title: str,
    matrix,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    value_suffix: str = "%",
) -> str:
    """Render a small labelled heatmap as an HTML table."""
    rows = as_nested_list(matrix)
    values = [value for row in rows for value in row]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 1.0
    head = "".join(f"<th>{html.escape(label)}</th>" for label in column_labels)
    body_rows = []
    for row_label, row in zip(row_labels, rows):
        cells = []
        for value in row:
            color = viridis_color(value, min_value, max_value)
            cells.append(
                f'<td style="background:{color}">'
                f"{html.escape(f'{value:.1f}{value_suffix}')}</td>"
            )
        body_rows.append(
            f"<tr><th>{html.escape(row_label)}</th>{''.join(cells)}</tr>"
        )
    return f"""
    <section class="panel">
      <h2>{html.escape(title)}</h2>
      <table class="heatmap">
        <thead><tr><th>Domain</th>{head}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </section>
    """


def domain_loss_table(before: dict[str, float], after: dict[str, float]) -> str:
    rows = []
    for domain in before:
        change = after[domain] - before[domain]
        direction = "good" if change < 0 else "bad"
        rows.append(
            "<tr>"
            f"<th>{html.escape(domain.capitalize())}</th>"
            f"<td>{before[domain]:.4f}</td>"
            f"<td>{after[domain]:.4f}</td>"
            f'<td class="{direction}">{change:+.4f}</td>'
            "</tr>"
        )
    return f"""
    <section class="panel">
      <h2>Per-domain held-out loss</h2>
      <table>
        <thead><tr><th>Domain</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


def token_routing_section(examples: list[dict]) -> str:
    max_expert = max(
        (int(item["expert"]) for example in examples for item in example["tokens"]),
        default=len(DEFAULT_EXPERT_DOMAINS) - 1,
    )
    legend_items = "".join(
        f'<span><i style="background:{EXPERT_COLORS[i % len(EXPERT_COLORS)]}"></i>'
        f"{html.escape(expert_label(i))}</span>"
        for i in range(max(max_expert + 1, len(DEFAULT_EXPERT_DOMAINS)))
    )
    cards = []
    for example in examples:
        target = DEFAULT_EXPERT_DOMAINS.get(example.get("domain_id"))
        domain = example["domain"].capitalize()
        target_label = ""
        if target:
            target_idx = example["domain_id"]
            target_label = (
                f'<span class="target">Target: '
                f"{html.escape(expert_label(target_idx))}</span>"
            )
        tokens = []
        for item in example["tokens"]:
            expert = int(item["expert"])
            color = EXPERT_COLORS[expert % len(EXPERT_COLORS)]
            token = item["token"].replace("\n", "\\n")
            tokens.append(
                f'<span class="token" style="background:{color}" '
                f'title="{html.escape(expert_label(expert))}">{html.escape(token)}</span>'
            )
        cards.append(
            f"""
            <article class="token-card">
              <h3>{html.escape(domain)} {target_label}</h3>
              <p>{html.escape(example["sentence"])}</p>
              <div class="tokens">{''.join(tokens)}</div>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <h2>Per-token routing examples</h2>
      <div class="legend">{legend_items}</div>
      <div class="token-grid">{''.join(cards)}</div>
    </section>
    """


def routing_dashboard_html(
    *,
    title: str,
    before_matrix,
    after_matrix,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    before_losses: dict[str, float],
    after_losses: dict[str, float],
    token_examples: list[dict],
    layer_idx: int,
    expert_domains: dict[int, str] | None = None,
) -> str:
    if expert_domains is None:
        expert_domains = DEFAULT_EXPERT_DOMAINS
    labelled_columns = [
        expert_label(i, expert_domains)
        for i, _ in enumerate(column_labels)
    ]
    body = "\n".join(
        [
            heatmap_table(
                "Before specialization: routing by domain",
                before_matrix,
                row_labels,
                labelled_columns,
            ),
            heatmap_table(
                "After specialization: routing by domain",
                after_matrix,
                row_labels,
                labelled_columns,
            ),
            domain_loss_table(before_losses, after_losses),
            token_routing_section(token_examples),
        ]
    )
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
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
    }}
    .meta {{
      color: #475569;
      margin: 8px 0 24px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 18px;
      margin: 0 0 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    h3 {{
      margin: 0 0 6px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 10px;
      text-align: center;
      font-size: 13px;
    }}
    th {{
      background: #f1f5f9;
      font-weight: 650;
    }}
    .heatmap td {{
      color: #ffffff;
      font-weight: 700;
      text-shadow: 0 1px 1px rgba(0, 0, 0, 0.35);
    }}
    .good {{
      color: #047857;
      font-weight: 700;
    }}
    .bad {{
      color: #b91c1c;
      font-weight: 700;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin: 0 0 14px;
      color: #334155;
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend i {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      display: inline-block;
    }}
    .token-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .token-card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px;
      background: #f8fafc;
    }}
    .token-card p {{
      color: #475569;
      margin: 0 0 10px;
      font-size: 13px;
      line-height: 1.35;
    }}
    .tokens {{
      line-height: 2;
    }}
    .token {{
      color: #ffffff;
      display: inline-block;
      margin: 1px;
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 13px;
      text-shadow: 0 1px 1px rgba(0, 0, 0, 0.35);
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p class="meta">Routing analyzed at decoder layer {layer_idx}.</p>
    {body}
  </main>
</body>
</html>
"""
