import collections.abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dacite
import yaml
from dacite import Config

from brainways_reg_model.utils.paths import PARAMS_PATH


@dataclass
class LabelParams:
    n_classes: int
    limits: Optional[Tuple[int, int]] = None
    label_names: Optional[List[str]] = None
    default: Optional[int] = None


@dataclass
class AtlasConfig:
    name: str
    brainglobe: bool
    exclude_regions: Optional[List[int]] = None
    axes: Tuple[int, int, int] = (0, 1, 2)


@dataclass
class ModelConfig:
    backbone: str
    outputs: List[str]


@dataclass
class DataConfig:
    atlas: AtlasConfig
    image_size: Tuple[int, int]
    batch_size: int
    label_params: Dict[str, LabelParams]
    structures: List[str] = field(default_factory=list)


@dataclass
class MonitorConfig:
    metric: str
    mode: str


@dataclass
class OptimizationConfig:
    train_bn: bool
    milestones: Tuple[int, int]
    lr: float
    lr_scheduler_gamma: float
    weight_decay: float
    max_epochs: int
    check_val_every_n_epoch: int
    monitor: MonitorConfig
    train_confidence: bool
    loss_weights: Dict[str, float]


@dataclass
class BrainwaysConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    opt: OptimizationConfig


@dataclass
class DuracellConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    opt: OptimizationConfig


def load_yaml(path: Union[Path, str]):
    with open(path) as fd:
        return yaml.safe_load(fd)


_CONFIGS = {}


def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_config(config_names: Union[str, Sequence[str]] = "default") -> BrainwaysConfig:
    global _CONFIGS
    if isinstance(config_names, str):
        config_names = (config_names,)
    config_names = tuple(config_names)
    if config_names not in _CONFIGS:
        all_configs = load_yaml(PARAMS_PATH)
        config = all_configs["default"]
        for config_name in config_names:
            if config_name != "default":
                update_config(config, all_configs[config_name])
        dacite_config = Config(cast=[Tuple])
        _CONFIGS[config_names] = dacite.from_dict(
            BrainwaysConfig, data=config, config=dacite_config
        )
    return _CONFIGS[config_names]


def load_synth_config():
    all_configs = load_yaml(PARAMS_PATH)
    return all_configs["synth"]
