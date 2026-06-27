import subprocess
import sys

from scripts.run_experiment import REPO_ROOT, resolve_experiment_script, tail_text


def test_resolve_experiment_script_accepts_numbered_scripts():
    path = resolve_experiment_script("05_train_moe_pretraining.py")

    assert path.name == "05_train_moe_pretraining.py"


def test_resolve_experiment_script_rejects_wrapper_script():
    try:
        resolve_experiment_script("run_experiment.py")
    except SystemExit as exc:
        assert "Unknown experiment script" in str(exc)
    else:
        raise AssertionError("run_experiment.py should not be accepted as an experiment")


def test_run_experiment_dry_run_local():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--backend",
            "local",
            "--dry-run",
            "05_train_moe_pretraining.py",
            "--",
            "--steps",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Local command: scripts/05_train_moe_pretraining.py --steps 1" in result.stdout


def test_tail_text_keeps_full_short_output():
    text, truncated = tail_text("hello\n", max_bytes=100)

    assert text == "hello\n"
    assert truncated is False


def test_tail_text_keeps_tail_for_long_output():
    text, truncated = tail_text("abcdef", max_bytes=3)

    assert text == "def"
    assert truncated is True
