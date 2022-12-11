from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from brainways.project.brainways_subject import BrainwaysSubject


@click.command()
@click.option(
    "--input",
    type=Path,
    required=True,
    help=(
        "Input directory of subject files to create registration model training data"
        " for."
    ),
)
@click.option(
    "--new-images-root",
    type=Path,
    required=True,
)
@click.option(
    "--old-images-root",
    type=Path,
)
def move_images_root(
    input: Path, new_images_root: Path, old_images_root: Optional[Path]
):
    paths = [p for p in input.glob("*") if p.is_dir()]
    for subject_path in tqdm(paths):
        print(subject_path)
        subject = BrainwaysSubject.open(subject_path)
        subject.move_images_root(
            new_images_root=new_images_root, old_images_root=old_images_root
        )
        subject.save(subject_path)
