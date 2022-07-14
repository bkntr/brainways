from pathlib import Path
from typing import Optional, Tuple

from aicsimageio.readers.czi_reader import CziReader
from PIL import Image
from tqdm import tqdm

from brainways.utils.image import resize_image, slice_to_uint8


def convert_czi_to_jpg(
    input_path: Path, output_path: Path, size: Optional[Tuple[int, int]] = None
):
    reader = CziReader(input_path)
    for scene_i, scene in enumerate(tqdm(reader.scenes)):
        reader.set_scene(scene)
        image_channels = reader.mosaic_data
        for image, channel_name in zip(image_channels, reader.channel_names):
            image = image.squeeze()
            if size is not None:
                image = resize_image(image, size, keep_aspect=True)
            image = slice_to_uint8(image)
            filename = f"{channel_name}_{input_path.stem}_scene_{scene_i}.jpg"
            # filename = f"{input_path.name} - Scene #{scene_i}.jpg"
            if not (output_path / channel_name).exists():
                (output_path / channel_name).mkdir(parents=True)
            Image.fromarray(image).save(str(output_path / channel_name / filename))
