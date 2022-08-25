from pathlib import Path

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
@click.option("--cells", type=Path, required=True, help="cells CSV file")
def import_cells(input: Path, cells: Path):
    paths = list(input.glob("*"))
    for project_path in tqdm(paths):
        project = BrainwaysProject.open(project_path)
        project.import_cells(cells)
        project.save(project_path)
