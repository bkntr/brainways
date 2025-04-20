from __future__ import annotations

from typing import Optional, Tuple

# TODO: remove kornia dependency
import kornia.geometry as KG
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from brainways.pipeline.brainways_params import AffineTransform2DParams
from brainways.transforms.base import BrainwaysTransform
from brainways.utils.image import ImageSizeHW


class BrainwaysAffineTransform2D(BrainwaysTransform):
    def __init__(
        self,
        params: Optional[AffineTransform2DParams] = None,
        mat: Optional[Tensor] = None,
        input_size: Optional[ImageSizeHW] = None,
        scale: Optional[float] = None,
    ):
        self.input_size = input_size
        self.scale = scale

        if (params is None) == (mat is None):
            raise ValueError("Please provide either mat or params")
        if params is not None:
            self.params = params
            center_x = params.cx
            center_y = params.cy

            if center_x is None:
                if self.input_size is None:
                    raise ValueError("Input size must be provided if params.cx is not")
                center_x = (self.input_size[1] - 1) // 2

            if center_y is None:
                if self.input_size is None:
                    raise ValueError("Input size must be provided if params.cy is not")
                center_y = (self.input_size[0] - 1) // 2

            scale_trans_mat = KG.get_affine_matrix2d(
                translations=torch.as_tensor(
                    [[params.tx, params.ty]], dtype=torch.float32
                ),
                center=torch.as_tensor([[center_x, center_y]], dtype=torch.float32),
                scale=torch.as_tensor([[params.sx, params.sy]], dtype=torch.float32),
                angle=torch.as_tensor([0], dtype=torch.float32),
            )
            rot_mat = KG.get_affine_matrix2d(
                translations=torch.as_tensor([[0, 0]], dtype=torch.float32),
                center=torch.as_tensor([[center_x, center_y]], dtype=torch.float32),
                scale=torch.as_tensor([[1, 1]], dtype=torch.float32),
                angle=torch.as_tensor([params.angle], dtype=torch.float32),
            )
            self.mat = scale_trans_mat @ rot_mat
        else:
            self.params = None
            self.mat = mat

        if scale is not None:
            scale_mat = KG.get_affine_matrix2d(
                translations=torch.as_tensor([[0, 0]], dtype=torch.float32),
                center=torch.as_tensor([[0, 0]], dtype=torch.float32),
                scale=torch.as_tensor([[scale, scale]], dtype=torch.float32),
                angle=torch.as_tensor([0], dtype=torch.float32),
            )
            self.mat = scale_mat @ self.mat

    def inv(self):
        return BrainwaysAffineTransform2D(
            mat=torch.inverse(self.mat),
            input_size=self.input_size,
            scale=1 / self.scale if self.scale is not None else None,
        )

    def transform_image(
        self,
        image: np.ndarray,
        output_size: ImageSizeHW | None = None,
        mode: str = "bilinear",
    ) -> np.ndarray:
        output_size = output_size or self.input_size
        warped = KG.warp_affine(
            to_tensor(image)[None],
            self.mat[:, :2],
            output_size,
            mode=mode,
        )

        channels = warped.shape[1]
        if channels == 1:
            return warped[0, 0].numpy()
        else:
            return warped[0].permute(1, 2, 0).numpy()

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """

        :param points: Nx2
        :return:
        """
        if len(points) > 0:
            return KG.transform_points(
                self.mat, torch.as_tensor(points).to(torch.float32)
            ).numpy()
        else:
            return np.empty(shape=(0, 2), dtype=points.dtype)

    @classmethod
    def find_transform_between_boxes(
        cls,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
        input_size: ImageSizeHW,
    ):
        """

        :param box1: x, y, w, h
        :param box2: x, y, w, h
        :return:
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        cx1 = x1 + w1 / 2
        cy1 = y1 + h1 / 2
        cx2 = x2 + w2 / 2
        cy2 = y2 + h2 / 2

        tx = cx2 - cx1
        ty = cy2 - cy1
        sx = w2 / w1
        sy = h2 / h1

        params = AffineTransform2DParams(
            angle=0, tx=tx, ty=ty, sx=sx, sy=sy, cx=cx1, cy=cy1
        )

        return cls(params=params, input_size=input_size)
