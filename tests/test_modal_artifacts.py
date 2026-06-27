import importlib.util
from pathlib import Path


def load_modal_artifacts_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "modal_artifacts.py"
    spec = importlib.util.spec_from_file_location("modal_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_destination_preserves_paths_for_multiple_downloads(tmp_path: Path):
    modal_artifacts = load_modal_artifacts_module()

    dest = modal_artifacts.local_destination(tmp_path, "nested/file.csv", multiple=True)

    assert dest == tmp_path / "nested" / "file.csv"


def test_local_destination_uses_dest_for_single_download(tmp_path: Path):
    modal_artifacts = load_modal_artifacts_module()

    dest = modal_artifacts.local_destination(tmp_path / "file.csv", "remote.csv", multiple=False)

    assert dest == tmp_path / "file.csv"


def test_modal_cmd_adds_env_when_present():
    modal_artifacts = load_modal_artifacts_module()

    cmd = modal_artifacts.modal_cmd("volume", "ls", "vol", env="dev")

    assert cmd[-2:] == ["--env", "dev"]
