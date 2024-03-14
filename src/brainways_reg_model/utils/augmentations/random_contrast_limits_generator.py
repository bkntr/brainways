from typing import Dict, Tuple, Union

import torch
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _common_param_check,
    _joint_range_check,
)
from kornia.utils.helpers import _extract_device_dtype
from torch.distributions import Uniform


class RandomContrastLimitsGenerator(RandomGeneratorBase):
    def __init__(
        self,
        min_limit: Union[torch.Tensor, Tuple[float, float]] = (0.0, 0.1),
        max_limit: Union[torch.Tensor, Tuple[float, float]] = (0.8, 1.0),
    ) -> None:
        super().__init__()
        self.min_limit = min_limit
        self.max_limit = max_limit

    def __repr__(self) -> str:
        repr = f"min_limit={self.min_limit}, max_limit={self.max_limit}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        min_limit = torch.as_tensor(self.min_limit, device=device, dtype=dtype)
        max_limit = torch.as_tensor(self.max_limit, device=device, dtype=dtype)

        _joint_range_check(min_limit, "min_limit", bounds=(0, 1))
        _joint_range_check(max_limit, "max_limit", bounds=(0, 1))

        self.min_limit_sampler = Uniform(
            min_limit[0], min_limit[1], validate_args=False
        )
        self.max_limit_sampler = Uniform(
            max_limit[0], max_limit[1], validate_args=False
        )
        self.uniform_sampler = Uniform(
            torch.tensor(0, device=device, dtype=dtype),
            torch.tensor(1, device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(
        self, batch_shape: torch.Size, same_on_batch: bool = False
    ) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.min_limit, self.max_limit])
        min_limit = _adapted_rsampling(
            (batch_size,), self.min_limit_sampler, same_on_batch
        )
        max_limit = _adapted_rsampling(
            (batch_size,), self.max_limit_sampler, same_on_batch
        )
        return dict(
            min_limit=min_limit.to(device=_device, dtype=_dtype),
            max_limit=max_limit.to(device=_device, dtype=_dtype),
        )
