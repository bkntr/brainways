from pathlib import Path

import click
from tqdm import tqdm

from brainways.project.brainways_subject import BrainwaysSubject
from brainways.utils.cell_detection_importer.keren_cell_detection_importer import (
    KerenCellDetectionsImporter,
)


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
    paths = list(p for p in input.glob("*") if p.is_dir())
    cell_detection_importer = KerenCellDetectionsImporter(
        cfos_threshold=cfos_threshold,
        drd1_threshold=drd1_threshold,
        drd2_threshold=drd2_threshold,
        oxtr_threshold=oxtr_threshold,
    )
    for subject_path in tqdm(paths):
        subject = BrainwaysSubject.open(subject_path)
        subject.import_cell_detections(
            root=cell_detections_root,
            cell_detection_importer=cell_detection_importer,
        )
        subject.save(subject_path)
