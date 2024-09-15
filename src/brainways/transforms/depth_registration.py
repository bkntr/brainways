from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import kornia.geometry as KG
import kornia.geometry.transform as KGT
import kornia.utils as KU
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from brainways.transforms.base import BrainwaysTransform
from brainways.utils.image import ImageSizeHW


@dataclass
class DepthRegistrationParams:
    td: float = 0.0
    rx: float = 0.0
    ry: float = 0.0


class DepthRegistration(BrainwaysTransform):
    def __init__(
        self,
        params: DepthRegistrationParams,
        volume_shape: Tuple[int, int, int],
        mode: str = "bilinear",
        _inverse: bool = False,
    ):
        self.params = params
        self.volume_shape = volume_shape
        self.mode = mode
        self.mat = KGT.get_affine_matrix3d(
            translations=torch.as_tensor([[0, 0, params.td]], dtype=torch.float32),
            center=torch.as_tensor(
                [
                    [
                        volume_shape[2] // 2,
                        volume_shape[1] // 2,
                        volume_shape[0] // 2,
                    ]
                ],
                dtype=torch.float32,
            ),
            scale=torch.as_tensor([[1, 1, 1]], dtype=torch.float32),
            angles=torch.as_tensor([[params.rx, params.ry, 0]], dtype=torch.float32),
        )
        self.mat_inverse = torch.inverse(self.mat)
        self._inverse = _inverse

    def inv(self) -> DepthRegistration:
        return DepthRegistration(
            params=self.params,
            volume_shape=self.volume_shape,
            mode=self.mode,
            _inverse=not self._inverse,
        )

    def slice_volume(self, volume: Tensor, mode: str | None = None) -> Tensor:
        depth, height, width = volume.shape
        grid = KU.create_meshgrid(
            height=height, width=width, normalized_coordinates=False
        )
        grid = grid.reshape(-1, 2)
        grid = self.transform_points(grid)
        grid = torch.as_tensor(grid).reshape(
            1, 1, volume.shape[1], volume.shape[2], 3
        )  # NDHW3
        # normalize grid for grid_sample
        grid = KG.normalize_pixel_coordinates3d(grid, depth, height, width)
        slice = F.grid_sample(
            volume[None, None],
            grid.float(),
            mode=mode or self.mode,
            align_corners=True,
        )
        return slice.squeeze()

    def transform_image(
        self,
        image: np.ndarray,
        output_size: ImageSizeHW | None = None,
        mode: str = "bilinear",
    ) -> np.ndarray:
        return image

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """

        :param points: Nx2
        :return: Nx3
        """
        if self._inverse:
            raise NotImplementedError("Inverse transform is not implemented")

        if len(points) > 0:
            points = torch.as_tensor(points)
            points_3d = torch.zeros(points.shape[0], 3, dtype=torch.float32)
            points_3d[:, :2] = points
            return KG.transform_points(self.mat_inverse, points_3d).numpy()
        else:
            return np.empty(shape=(0, 3), dtype=points.dtype)
