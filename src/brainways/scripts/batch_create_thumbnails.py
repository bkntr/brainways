import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import click
from aicsimageio import AICSImage
from PIL import Image
from tqdm import tqdm

from brainways.utils.image import resize_image, slice_to_uint8
from brainways.utils.io_utils import ImagePath


def creat_thumbnail(
    input: Path,
    output: Path,
    size: Optional[Tuple[int, int]] = None,
    channels: Optional[List[int]] = None,
):
    reader = AICSImage(input)
    if channels is None:
        channels = range(reader.channel_names)
    for scene_i, scene in enumerate(tqdm(reader.scenes)):
        reader.set_scene(scene)
        for channel_i in channels:
            channel_name = reader.channel_names[channel_i]
            image = reader.get_image_data("YX", C=channel_i).squeeze()
            if size is not None:
                image = resize_image(image, size, keep_aspect=True)
            image = slice_to_uint8(image)
            lowres_filename = (
                Path(str(ImagePath(str(input), scene=scene_i, channel=channel_i))).name
                + ".jpg"
            )
            if not (output / channel_name).exists():
                (output / channel_name).mkdir(parents=True)
            Image.fromarray(image).save(str(output / channel_name / lowres_filename))


@click.command()
@click.option(
    "--input",
    type=Path,
    required=True,
    help="Input directory of images to create thumbnails for.",
)
@click.option(
    "--output",
    type=Path,
    required=True,
    help="Results will be written to this directory.",
)
@click.option(
    "--pattern",
    type=str,
    default="*",
    help="Only create thumbnail for file names matching this pattern",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Look for images in input directory recursively.",
)
@click.option(
    "--size",
    type=int,
    nargs=2,
    default=(1024, 1024),
    help="Thumbnail image size.",
)
@click.option(
    "-c",
    "--channel",
    type=int,
    multiple=True,
    help="Channel index in the image (optional).",
)
def batch_create_thumbnails(
    input: Path,
    output: Path,
    pattern: str,
    recursive: bool,
    size: Optional[Tuple[int, int]] = None,
    channel: Optional[List[int]] = None,
):
    input_path = Path(input)
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)

    if input_path.is_file():
        input_paths = [input_path]
    else:
        if recursive:
            input_paths = [p for p in input_path.rglob(pattern) if p.is_file()]
        else:
            input_paths = [p for p in input_path.glob(pattern) if p.is_file()]

    for p in tqdm(input_paths):
        try:
            print(p)
            if len(list(output_path.rglob(f"{p.name}*"))) == 0:
                creat_thumbnail(
                    input=p, output=output_path, size=size, channels=channel
                )
        except KeyboardInterrupt:
            raise
        except Exception:
            print(f"{p} failed:")
            traceback.print_exc()
