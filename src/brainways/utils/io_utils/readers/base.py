from abc import ABC
from pathlib import Path
from typing import Optional, Tuple

from brainways.utils.image import Box, ImageSizeHW


class ImageReader(ABC):
    def __init__(self, path: Path, scene: int):
        self.path = path
        self.scene = scene
        self.scene_bb: Optional[Box] = None

    def read_image(
        self,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        channel: int = 0,
        size: Optional[ImageSizeHW] = None,
        scale: Optional[float] = None,
    ):
        ...

    @property
    def size(self) -> ImageSizeHW:
        return int(self.scene_bb[3]), int(self.scene_bb[2])
