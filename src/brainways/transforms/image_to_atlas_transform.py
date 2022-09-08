from __future__ import annotations

from typing import List

import numpy as np

from brainways.transforms.affine_transform_2d import BrainwaysAffineTransform2D
from brainways.transforms.base import BrainwaysTransform
from brainways.transforms.depth_registration import DepthRegistration
from brainways.transforms.tps_transform import TPSTransform
from brainways.utils.image import ImageSizeHW


class ImageToAtlasTransform(BrainwaysTransform):
    def __init__(
        self,
        atlas_transform: DepthRegistration | None,
        affine_2d_transform: BrainwaysAffineTransform2D | None,
        tps_transform: TPSTransform | None,
        inverse: bool = False,
    ):
        self.atlas_transform = atlas_transform
        self.affine_2d_transform = affine_2d_transform
        self.tps_transform = tps_transform
        self.inverse = inverse

    @property
    def transforms(self) -> List[BrainwaysTransform]:
        transforms = [
            self.affine_2d_transform,
            self.tps_transform,
            self.atlas_transform,
        ]
        if self.inverse:
            transforms = transforms[::-1]
        transforms = [transform for transform in transforms if transform is not None]
        return transforms

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

    def inv(self) -> ImageToAtlasTransform:
        return ImageToAtlasTransform(
            atlas_transform=self.atlas_transform,
            affine_2d_transform=self.affine_2d_transform.inv()
            if self.affine_2d_transform is not None
            else None,
            tps_transform=self.tps_transform.inv()
            if self.tps_transform is not None
            else None,
            inverse=not self.inverse,
        )
