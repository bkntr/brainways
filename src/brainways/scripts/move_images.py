from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject


@click.command()
@click.option(
    "--project",
    type=Path,
    required=True,
    help="Brainways project path.",
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
    project: Path, new_images_root: Path, old_images_root: Optional[Path]
):
    project = BrainwaysProject.open(project)
    for subject in tqdm(project.subjects):
        subject.move_images_root(
            new_images_root=new_images_root, old_images_root=old_images_root
        )
        subject.save()
