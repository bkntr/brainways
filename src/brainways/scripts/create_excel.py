from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import ExcelMode


@click.command()
@click.option(
    "--project",
    "project_path",
    type=Path,
    required=True,
    help="Brainways project path.",
)
@click.option(
    "--output",
    "output",
    type=Path,
    required=True,
)
@click.option(
    "--min-region-area-um2",
    type=int,
    default=250,
    show_default=True,
)
@click.option(
    "--cells-per-area-um2",
    type=int,
    default=250,
    show_default=True,
)
@click.option(
    "--min-cell-size-um",
    type=int,
)
@click.option(
    "--max-cell-size-um",
    type=int,
)
@click.option(
    "--excel-mode",
    type=ExcelMode,
    default=ExcelMode.ROW_PER_SUBJECT,
    show_default=True,
)
def create_excel(
    project_path: Path,
    output: Path,
    min_region_area_um2: Optional[int],
    cells_per_area_um2: Optional[int],
    min_cell_size_um: Optional[float],
    max_cell_size_um: Optional[float],
    excel_mode: ExcelMode,
):
    project = BrainwaysProject.open(project_path)
    for _ in tqdm(
        project.calculate_results_iter(
            path=output,
            min_region_area_um2=min_region_area_um2,
            cells_per_area_um2=cells_per_area_um2,
            min_cell_size_um=min_cell_size_um,
            max_cell_size_um=max_cell_size_um,
            excel_mode=excel_mode,
        ),
        total=len(project.subjects),
    ):
        ...


if __name__ == "__main__":
    create_excel()
