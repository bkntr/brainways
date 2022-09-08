from pathlib import Path
from typing import Optional, Tuple

import aicsimageio

from brainways.utils.image import ImageSizeHW, resize_image
from brainways.utils.io_utils.readers.base import ImageReader


class AicsImageIoReader(ImageReader):
    def __init__(self, path: Path, scene: Optional[int]):
        super().__init__(path, scene)
        self.aics_image = aicsimageio.AICSImage(path)
        if scene is not None:
            self.aics_image.set_scene(scene)
        self.scene_bb = (0, 0, self.aics_image.dims.X, self.aics_image.dims.Y)

    def read_image(
        self,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        channel: int = 0,
        size: Optional[ImageSizeHW] = None,
        scale: Optional[float] = None,
    ):
        image = self.aics_image.get_image_data("YX", C=channel).squeeze()

        # read part of the image
        if bounding_box is not None:
            x, y, w, h = bounding_box
            scene_x, scene_y, scene_w, scene_h = self.scene_bb
            x = int(round(x * scene_w))
            y = int(round(y * scene_h))
            w = int(round(w * scene_w))
            h = int(round(h * scene_h))
        else:
            x, y, w, h = self.scene_bb

        image = image[y : y + h + 1, x : x + w + 1]
        if size is not None or scale is not None:
            image = resize_image(image, size=size, scale=scale)
        return image
