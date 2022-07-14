from __future__ import annotations

import numpy as np

from brainways.utils.image import ImageSizeHW


class BrainwaysTransform:
    def transform_image(
        self,
        image: np.ndarray,
        output_size: ImageSizeHW | None = None,
        mode: str = "bilinear",
    ) -> np.ndarray:
        raise NotImplementedError()

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
