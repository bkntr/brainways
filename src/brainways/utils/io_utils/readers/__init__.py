from pathlib import Path
from typing import List, Union

from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils.image_path import ImagePath
from brainways.utils.io_utils.readers.qupath_reader import QupathReader


def get_scenes(filename: Union[str, Path]):
    reader = QupathReader(filename)
    return reader.scenes


def get_image_size(path: ImagePath) -> ImageSizeHW:
    reader = QupathReader(path.filename)
    if path.scene is not None:
        reader.set_scene(path.scene)
    return (reader.dims.Y, reader.dims.X)


def get_channels(filename: Union[str, Path]) -> List[str]:
    reader = QupathReader(filename)
    channel_names = reader.channel_names
    assert channel_names is not None, f"Channel names not found in {filename}"
    return channel_names
