from pathlib import Path

import click
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject
from brainways.utils.cell_detection_importer.keren_cell_detection_importer import (
    KerenCellDetectionsImporter,
)


@click.command()
@click.option(
    "--project-path",
    type=Path,
    required=True,
    help="Input project.",
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
    project_path: Path,
    cell_detections_root: Path,
    cfos_threshold: int,
    drd1_threshold: int,
    drd2_threshold: int,
    oxtr_threshold: int,
):
    project = BrainwaysProject.open(project_path, lazy_init=True)
    cell_detection_importer = KerenCellDetectionsImporter(
        cfos_threshold=cfos_threshold,
        drd1_threshold=drd1_threshold,
        drd2_threshold=drd2_threshold,
        oxtr_threshold=oxtr_threshold,
    )
    progress = project.import_cell_detections_iter(
        importer=cell_detection_importer, cell_detections_root=cell_detections_root
    )
    for _ in tqdm(progress, total=project.n_valid_images):
        pass
