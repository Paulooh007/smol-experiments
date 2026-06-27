"""Run one experiment script locally or on Modal.

Local is the default and remains the cheapest path. Modal is explicit:

    uv run --group modal python scripts/run_experiment.py \
        --backend modal --gpu T4 scripts/05_train_moe_pretraining.py -- \
        --steps 100 --max-samples 1000

Use ``--backend auto`` when you want Modal if it is installed/configured, with
a local fallback otherwise. Use ``--dry-run`` to print the resolved command
without launching anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
REMOTE_WORKDIR = "/root/smol-experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("local", "modal", "auto"),
        default="local",
        help="Where to run the experiment. 'auto' tries Modal, then falls back locally.",
    )
    parser.add_argument("--gpu", default="T4", help="Modal GPU type, e.g. T4, L4, A10G, A100.")
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60)
    parser.add_argument("--modal-app", default="smol-experiments")
    parser.add_argument("--modal-volume", default="smol-experiments-outputs")
    parser.add_argument(
        "--max-return-log-bytes",
        type=int,
        default=200_000,
        help="Maximum remote log bytes returned to the local terminal after completion.",
    )
    parser.add_argument(
        "--modal-secret",
        action="append",
        default=[],
        help="Modal Secret name to attach to the remote run. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("script", help="Numbered experiment script under scripts/.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.script_args and args.script_args[0] == "--":
        args.script_args = args.script_args[1:]
    return args


def resolve_experiment_script(script: str) -> Path:
    raw = Path(script)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((REPO_ROOT / raw).resolve())
        candidates.append((SCRIPTS_DIR / raw).resolve())

    allowed = {
        path.resolve()
        for path in SCRIPTS_DIR.glob("*.py")
        if path.name != Path(__file__).name and path.name[0].isdigit()
    }
    for candidate in candidates:
        if candidate in allowed:
            return candidate

    choices = ", ".join(sorted(path.name for path in allowed))
    raise SystemExit(f"Unknown experiment script {script!r}. Choose one of: {choices}")


def project_dependencies() -> list[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return list(data["project"]["dependencies"])


def command_for_display(script_path: Path, script_args: list[str]) -> str:
    rel_script = script_path.relative_to(REPO_ROOT)
    parts = [str(rel_script), *script_args]
    return " ".join(parts)


def run_local(script_path: Path, script_args: list[str], dry_run: bool) -> int:
    cmd = [sys.executable, str(script_path), *script_args]
    print(f"Local command: {command_for_display(script_path, script_args)}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def modal_available() -> bool:
    try:
        import modal  # noqa: F401
    except ImportError:
        return False
    return True


def tail_text(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if max_bytes <= 0 or len(encoded) <= max_bytes:
        return text, False
    trimmed = encoded[-max_bytes:].decode("utf-8", errors="replace")
    return trimmed, True


def print_remote_result(result: dict, max_return_log_bytes: int) -> int:
    output = result.get("output", "")
    visible, truncated = tail_text(output, max_return_log_bytes)
    if visible:
        print("\n================ Remote output ================")
        if truncated:
            print(f"[showing last {max_return_log_bytes:,} bytes of remote output]\n")
        print(visible, end="" if visible.endswith("\n") else "\n")
        print("================ End remote output ================")
    return int(result["returncode"])


def run_modal(script_path: Path, script_args: list[str], args: argparse.Namespace) -> int:
    try:
        import modal
    except ImportError as exc:
        raise SystemExit(
            "Modal is not installed. Run `uv sync --group dev --group modal` "
            "or invoke this wrapper with `uv run --group modal ...`."
        ) from exc

    rel_script = str(script_path.relative_to(REPO_ROOT))
    print(f"Modal command: {command_for_display(script_path, script_args)}")
    print(f"Modal GPU: {args.gpu} | volume: {args.modal_volume} | app: {args.modal_app}")
    if args.modal_secret:
        print(f"Modal secrets: {', '.join(args.modal_secret)}")
    if args.dry_run:
        return 0

    deps = project_dependencies()
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*deps)
        .workdir(REMOTE_WORKDIR)
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": f"{REMOTE_WORKDIR}/outputs/.cache/huggingface",
                "HF_DATASETS_CACHE": f"{REMOTE_WORKDIR}/outputs/.cache/huggingface/datasets",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        .add_local_dir(
            str(REPO_ROOT / "src"),
            remote_path=f"{REMOTE_WORKDIR}/src",
        )
        .add_local_dir(
            str(REPO_ROOT / "scripts"),
            remote_path=f"{REMOTE_WORKDIR}/scripts",
        )
        .add_local_file(
            str(REPO_ROOT / "pyproject.toml"),
            remote_path=f"{REMOTE_WORKDIR}/pyproject.toml",
        )
    )
    app = modal.App(args.modal_app)
    volume = modal.Volume.from_name(args.modal_volume, create_if_missing=True)
    secrets = [modal.Secret.from_name(name) for name in args.modal_secret]

    @app.function(
        image=image,
        gpu=args.gpu,
        timeout=args.timeout,
        volumes={f"{REMOTE_WORKDIR}/outputs": volume},
        secrets=secrets,
        serialized=True,
    )
    def _run_remote(script: str, remote_args: list[str]) -> dict:
        cmd = [sys.executable, script, *remote_args]
        print(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=REMOTE_WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_parts = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_parts.append(line)
        returncode = process.wait()
        volume.commit()
        return {"returncode": returncode, "output": "".join(output_parts)}

    with app.run():
        result = _run_remote.remote(rel_script, script_args)
    return print_remote_result(result, args.max_return_log_bytes)


def main() -> None:
    args = parse_args()
    script_path = resolve_experiment_script(args.script)

    if args.backend == "local":
        raise SystemExit(run_local(script_path, args.script_args, args.dry_run))

    if args.backend == "modal":
        raise SystemExit(run_modal(script_path, args.script_args, args))

    if modal_available():
        try:
            raise SystemExit(run_modal(script_path, args.script_args, args))
        except Exception as exc:
            print(f"Modal run unavailable ({exc}); falling back to local.")
    raise SystemExit(run_local(script_path, args.script_args, args.dry_run))


if __name__ == "__main__":
    main()
