from typing import List

from torch import Tensor

from brainways.transforms.base import BrainwaysTransform


class Compose(BrainwaysTransform):
    def __init__(
        self,
        transforms_2d: List[BrainwaysTransform],
        transforms_3d: List[BrainwaysTransform],
    ):
        self._transforms_2d = transforms_2d
        self._transforms_3d = transforms_3d

    def transform_points(self, points: Tensor) -> Tensor:
        for t in self._transforms_2d:
            points = t.transform_points(points)
        for t in self._transforms_3d:
            points = t.transform_points(points)
        return points
