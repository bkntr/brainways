from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject


@click.command()
@click.option(
    "--input",
    type=Path,
    required=True,
    help=(
        "Input directory of project files to create registration model training data"
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
    paths = list(input.glob("*"))
    for project_path in tqdm(paths):
        print(project_path)
        project = BrainwaysProject.open(project_path)
        project.move_images_root(
            new_images_root=new_images_root, old_images_root=old_images_root
        )
        project.save(project_path)
