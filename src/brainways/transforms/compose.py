from __future__ import annotations

from typing import List

import numpy as np

from brainways.transforms.base import BrainwaysTransform
from brainways.utils.image import ImageSizeHW


class Compose(BrainwaysTransform):
    def __init__(self, transforms: List[BrainwaysTransform]):
        self.transforms = transforms

    def transform_image(
        self,
        image: np.ndarray,
        output_size: ImageSizeHW | None = None,
        mode: str = "bilinear",
    ) -> np.ndarray:
        for transform in self.transforms:
            image = transform.transform_image(image, output_size=output_size, mode=mode)

        return image

    def transform_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        for transform in self.transforms:
            points = transform.transform_points(points)
        return points

    def inv(self) -> Compose:
        return Compose(
            transforms=[transform.inv() for transform in self.transforms[::-1]]
        )
