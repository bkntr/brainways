import os
import platform
import subprocess
from pathlib import Path

_BRAINWAYS_PATH = Path.home() / ".brainways"


def get_brainways_dir() -> Path:
    if not _BRAINWAYS_PATH.exists():
        _BRAINWAYS_PATH.mkdir()
    return _BRAINWAYS_PATH


def get_brainways_config_path():
    return _BRAINWAYS_PATH / "brainways.toml"


def open_directory(path: Path):
    if not path.is_dir():
        path = path.parent
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    assert path.is_dir(), f"Path is not a directory: {path}"

    system = platform.system()

    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":  # macOS
        subprocess.run(["open", path])
    elif system == "Linux":
        subprocess.run(["xdg-open", path])
    else:
        raise OSError(f"Unsupported operating system: {system}")
