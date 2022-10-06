from pathlib import Path

import click
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject
from brainways.utils.cell_detection_importer.utils import (
    cell_detection_importer_types,
    get_cell_detection_importer,
)


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
    "--cell-detections-root",
    type=Path,
    required=True,
    help="cell detection files root directory",
)
@click.option(
    "--type",
    type=click.Choice(cell_detection_importer_types(), case_sensitive=False),
    required=True,
    help="cell detection type",
)
def import_cell_detections(input: Path, cell_detections_root: Path, type: str):
    paths = list(input.glob("*"))
    cell_detection_importer = get_cell_detection_importer(type)
    for project_path in tqdm(paths):
        project = BrainwaysProject.open(project_path)
        project.import_cell_detections(
            root=cell_detections_root, cell_detection_importer=cell_detection_importer
        )
        project.save(project_path)
