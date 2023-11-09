from pathlib import Path

_BRAINWAYS_PATH = Path.home() / ".brainways"


def get_brainways_dir() -> Path:
    if not _BRAINWAYS_PATH.exists():
        _BRAINWAYS_PATH.mkdir()
    return _BRAINWAYS_PATH


def get_brainways_config_path():
    return _BRAINWAYS_PATH / "brainways.toml"
