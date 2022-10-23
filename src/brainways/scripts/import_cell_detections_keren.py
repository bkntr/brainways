from pathlib import Path

import click
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject
from brainways.utils.cell_detection_importer.qupath_cell_detection_importer import (
    KerenCellDetectionsImporter,
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
    "--cfos-threshold",
    type=int,
    required=True,
)
@click.option(
    "--drd1-threshold",
    type=int,
    required=True,
)
@click.option(
    "--drd2-threshold",
    type=int,
    required=True,
)
@click.option(
    "--oxtr-threshold",
    type=int,
    required=True,
)
def import_cell_detections_keren(
    input: Path,
    cell_detections_root: Path,
    cfos_threshold: int,
    drd1_threshold: int,
    drd2_threshold: int,
    oxtr_threshold: int,
):
    paths = list(input.glob("*"))
    cell_detection_importer = KerenCellDetectionsImporter()
    for project_path in tqdm(paths):
        project = BrainwaysProject.open(project_path)
        project.import_cell_detections(
            root=cell_detections_root,
            cell_detection_importer=cell_detection_importer,
            cfos_threshold=cfos_threshold,
            drd1_threshold=drd1_threshold,
            drd2_threshold=drd2_threshold,
            oxtr_threshold=oxtr_threshold,
        )
        project.save(project_path)
