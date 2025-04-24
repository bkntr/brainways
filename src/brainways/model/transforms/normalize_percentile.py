from __future__ import annotations

from typing import Any

import numpy as np
from albumentations import ImageOnlyTransform  # type: ignore[import-untyped]


class NormalizePercentile(ImageOnlyTransform):
    def __init__(
        self,
        limits: tuple[float, float],
        p: float = 1.0,
    ):
        super().__init__(p=p)
        if not (0 <= limits[0] < limits[1] <= 100):
            raise ValueError(
                f"Error during NormalizePercentile initialization. Percentile values should be in the range [0, 100]. "
                f"Got: {limits}."
            )
        self.limits = np.array(limits, dtype=np.float32)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        lower, upper = np.percentile(img, self.limits)
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower + 1e-6)
        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("limits",)
