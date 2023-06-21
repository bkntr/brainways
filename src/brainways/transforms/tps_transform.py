from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from brainways.pipeline.brainways_params import TPSTransformParams
from brainways.transforms.base import BrainwaysTransform
from brainways.utils.image import ImageSizeHW


class TPSTransform(BrainwaysTransform):
    def __init__(self, params: TPSTransformParams, scale: Optional[float] = None):
        self.scale = scale

        scale = scale or 1.0
        self.params = params
        matches = [cv2.DMatch(i, i, 0) for i in range(len(self.params.points_src))]
        self.tps_image = cv2.createThinPlateSplineShapeTransformer()

        points_src_np = np.array(self.params.points_src, dtype=np.float32)[None]
        points_dst_np = np.array(self.params.points_dst, dtype=np.float32)[None]

        self.tps_image.estimateTransformation(
            points_dst_np * scale, points_src_np * scale, matches
        )
        self.tps_points = cv2.createThinPlateSplineShapeTransformer()
        self.tps_points.estimateTransformation(
            points_src_np * scale, points_dst_np * scale, matches
        )

    def inv(self):
        params_inv = TPSTransformParams(
            points_src=self.params.points_dst,
            points_dst=self.params.points_src,
        )
        return TPSTransform(params=params_inv, scale=self.scale)

    def transform_image(
        self,
        image: np.ndarray,
        output_size: ImageSizeHW | None = None,
        mode: str = "bilinear",
    ) -> np.ndarray:
        if mode == "bilinear":
            mode_cv2 = cv2.INTER_LINEAR
        elif mode == "nearest":
            mode_cv2 = cv2.INTER_NEAREST
        else:
            raise ValueError(mode)
        out_img = self.tps_image.warpImage(np.array(image), flags=mode_cv2)
        return out_img

    def transform_points(self, points: np.array) -> np.array:
        """

        :param points: [[x1, y1], ...] Nx2
        :return:
        """
        if len(points) > 0:
            _, transformed_points = self.tps_points.applyTransformation(
                np.array(points)[None].astype(np.float32)
            )
            return transformed_points[0]
        else:
            return np.empty(shape=(0, 2), dtype=points.dtype)
