from __future__ import annotations

import numpy as np

from brainways.transforms.base import BrainwaysTransform
from brainways.utils.image import ImageSizeHW


class IdentityTransform(BrainwaysTransform):
    def __init__(self):
        pass

    def transform_image(
        self,
        image: np.ndarray,
        output_size: ImageSizeHW | None = None,
        mode: str = "bilinear",
    ) -> np.ndarray:
        return image

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return points

    def inv(self):
        return IdentityTransform()
