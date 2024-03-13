from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from torch import Tensor

from brainways_reg_model.utils.augmentations.random_contrast_limits_generator import (
    RandomContrastLimitsGenerator,
)


class RandomContrastLimits(IntensityAugmentationBase2D):
    def __init__(
        self,
        min_limit: Union[Tensor, Tuple[float, float]] = (0.0, 0.1),
        max_limit: Union[Tensor, Tuple[float, float]] = (0.8, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
        )
        self.min_limit = min_limit
        self.max_limit = max_limit
        self._param_generator = cast(
            RandomContrastLimitsGenerator,
            RandomContrastLimitsGenerator(min_limit, max_limit),
        )

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        min_values = torch.quantile(
            input.flatten(), params["min_limit"].to(input.device)
        )[:, None, None, None]
        max_values = torch.quantile(
            input.flatten(), params["max_limit"].to(input.device)
        )[:, None, None, None]
        normalized = (input - min_values) / (max_values - min_values)
        normalized = torch.clip(normalized, 0, 1)
        return normalized
