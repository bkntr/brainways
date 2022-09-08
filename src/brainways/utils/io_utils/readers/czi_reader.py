from pathlib import Path
from typing import Optional, Tuple

import aicspylibczi

from brainways.utils.image import ImageSizeHW, resize_image
from brainways.utils.io_utils.readers.base import ImageReader


class CziReader(ImageReader):
    def __init__(self, path: Path, scene: int):
        super().__init__(path, scene)
        self.czi_file = aicspylibczi.CziFile(path)

        bb = self.czi_file.get_scene_bounding_box(self.scene)
        self.scene_bb = (bb.x, bb.y, bb.w, bb.h)

    def read_image(
        self,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        channel: int = 0,
        size: Optional[ImageSizeHW] = None,
        scale: Optional[float] = None,
    ):
        # read part of the image
        if bounding_box is not None:
            x, y, w, h = bounding_box
            scene_x, scene_y, scene_w, scene_h = self.scene_bb
            bb_x = scene_x + int(round(x * scene_w))
            bb_y = scene_y + int(round(y * scene_h))
            bb_w = int(round(w * scene_w))
            bb_h = int(round(h * scene_h))
            bb = (bb_x, bb_y, bb_w, bb_h)
        else:
            bb = self.scene_bb

        image = self.czi_file.read_mosaic(bb, C=channel).squeeze()
        if size is not None or scale is not None:
            image = resize_image(image, size=size, scale=scale)

        return image
