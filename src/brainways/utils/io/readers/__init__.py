from pathlib import Path
from typing import Union

import aicsimageio

from brainways.utils.image import ImageSizeHW
from brainways.utils.io.image_path import ImagePath
from brainways.utils.io.readers.aicsimageio_reader import AicsImageIoReader
from brainways.utils.io.readers.czi_reader import CziReader


def get_reader(path: ImagePath):
    filename = Path(path.filename)
    if filename.suffix == ".czi":
        return CziReader(path=filename, scene=path.scene)
    else:
        return AicsImageIoReader(path=filename, scene=path.scene)


def get_scenes(filename: Union[str, Path]):
    aics_image = aicsimageio.AICSImage(filename)
    return aics_image.scenes


def get_image_size(path: ImagePath) -> ImageSizeHW:
    aics_image = aicsimageio.AICSImage(path.filename)
    if path.scene is not None:
        aics_image.set_scene(path.scene)
    return (aics_image.dims.Y, aics_image.dims.X)


def get_channels(filename: Union[str, Path]):
    aics_image = aicsimageio.AICSImage(filename)
    return aics_image.channel_names
