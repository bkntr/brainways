from pathlib import Path

_BRAINWAYS_PATH = Path.home() / ".brainways"


def get_brainways_dir() -> Path:
    return _BRAINWAYS_PATH
