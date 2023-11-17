from dataclasses import asdict, dataclass

import toml

from brainways.utils.paths import get_brainways_config_path


@dataclass
class BrainwaysConfig:
    initialized: bool = False


def write_config(config: BrainwaysConfig) -> None:
    config_path = get_brainways_config_path()
    with open(config_path, "w") as f:
        toml.dump(asdict(config), f)


def load_config() -> BrainwaysConfig:
    config_path = get_brainways_config_path()
    if not config_path.exists():
        write_config(BrainwaysConfig())
    config = toml.load(config_path)
    return BrainwaysConfig(**config)
