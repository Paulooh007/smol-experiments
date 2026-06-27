"""List and download artifacts from the Modal experiment volume.

Examples:

    uv run --group modal python scripts/modal_artifacts.py list
    uv run --group modal python scripts/modal_artifacts.py pull
    uv run --group modal python scripts/modal_artifacts.py pull training_dashboard.html

The default ``pull`` command downloads the common outputs produced by the MoE
track when they exist: checkpoints, metric CSVs, and the dashboard HTML.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_VOLUME = "smol-experiments-outputs"
DEFAULT_DEST = Path("artifacts/modal")
DEFAULT_ARTIFACTS = [
    "moe_upcycled.pt",
    "moe_pretrained.pt",
    "moe_specialized.pt",
    "moe_pretraining_metrics.csv",
    "moe_specialization_metrics.csv",
    "training_dashboard.html",
    "routing_analysis/routing_dashboard.html",
    "routing_analysis/routing_before.csv",
    "routing_analysis/routing_after.csv",
    "routing_analysis/domain_losses.csv",
    "routing_analysis/token_routing.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--env", default=None, help="Modal environment name.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List files in the Modal volume.")
    list_parser.add_argument("path", nargs="?", default="/")
    list_parser.add_argument("--json", action="store_true", help="Emit Modal's JSON listing.")

    pull_parser = subparsers.add_parser("pull", help="Download files from the Modal volume.")
    pull_parser.add_argument(
        "paths",
        nargs="*",
        help="Remote paths to download. Defaults to common experiment artifacts.",
    )
    pull_parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    pull_parser.add_argument("--force", action="store_true", help="Overwrite local files.")
    pull_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested/default artifact is missing.",
    )
    return parser.parse_args()


def modal_cmd(*args: str, env: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "modal", *args]
    if env is not None:
        cmd.extend(["--env", env])
    return cmd


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, check=check)


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def list_volume(volume: str, path: str, emit_json: bool, env: str | None) -> int:
    cmd = modal_cmd("volume", "ls", volume, path, env=env)
    if emit_json:
        cmd.append("--json")
    return run(cmd).returncode


def local_destination(dest: Path, remote_path: str, multiple: bool) -> Path:
    cleaned = remote_path.lstrip("/") or "volume"
    if multiple:
        return dest / cleaned
    return dest


def pull_one(
    volume: str,
    remote_path: str,
    dest: Path,
    force: bool,
    strict: bool,
    multiple: bool,
    env: str | None,
) -> int:
    local_path = local_destination(dest, remote_path, multiple)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = modal_cmd("volume", "get", volume, remote_path, str(local_path), env=env)
    if force:
        cmd.append("--force")
    print(f"Downloading {volume}:{remote_path} -> {local_path}")
    result = run_capture(cmd)
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="")
        return 0

    message = (result.stderr or result.stdout or "").strip()
    prefix = "ERROR" if strict else "warning"
    print(f"{prefix}: could not download {remote_path}: {message}", file=sys.stderr)
    return result.returncode if strict else 0


def pull_artifacts(args: argparse.Namespace) -> int:
    paths = args.paths or DEFAULT_ARTIFACTS
    multiple = len(paths) > 1
    args.dest.mkdir(parents=True, exist_ok=True)
    status = 0
    for remote_path in paths:
        code = pull_one(
            volume=args.volume,
            remote_path=remote_path,
            dest=args.dest,
            force=args.force,
            strict=args.strict,
            multiple=multiple,
            env=args.env,
        )
        status = status or code
    print(f"Artifacts destination: {args.dest.resolve()}")
    return status


def main() -> None:
    args = parse_args()
    if args.command == "list":
        raise SystemExit(list_volume(args.volume, args.path, args.json, args.env))
    if args.command == "pull":
        raise SystemExit(pull_artifacts(args))
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
