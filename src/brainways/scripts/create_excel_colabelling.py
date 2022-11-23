import logging
from pathlib import Path

import click
import pandas as pd
from pandas import ExcelWriter
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject
from brainways.utils._imports import NAPARI_AVAILABLE
from brainways.utils.cells import get_cell_struct_ids, get_struct_colors

if NAPARI_AVAILABLE:
    import napari


def display_cells_3d(project: BrainwaysProject):
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Please install napari to display results: "
            "`pip install napari` or `pip install brainways[all]`"
        ) from None

    all_cells = project.get_cells_on_atlas()
    struct_ids = get_cell_struct_ids(all_cells, project.atlas.brainglobe_atlas)
    colors = get_struct_colors(struct_ids, project.atlas.brainglobe_atlas)

    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3
    viewer.add_image(project.atlas.reference.numpy(), name="Atlas")
    viewer.add_points(
        all_cells[["z", "y", "x"]].values,
        name="Cells",
        ndim=3,
        size=1,
        edge_color=colors,
        face_color=colors,
    )
    viewer.title = project.project_path.name
    napari.run()


def display_cells_2d(project: BrainwaysProject):
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Please install napari to display results: "
            "`pip install napari` or `pip install brainways[all]`"
        ) from None

    for _, document in project.valid_documents:
        if not project.cell_detections_path(document.path).exists():
            logging.warning(
                f"{document.path}: missing cells, please run cell detection."
            )
            continue

        viewer = napari.Viewer()
        image = project.read_highres_image(document, level=1)
        viewer.add_image(image, name=str(document.path))

        cells_atlas = project.get_cells_on_atlas([document])
        cells = project.get_valid_cells(document)
        struct_ids = get_cell_struct_ids(cells_atlas, project.atlas.brainglobe_atlas)
        colors = get_struct_colors(struct_ids, project.atlas.brainglobe_atlas)

        viewer.add_points(
            cells[["y", "x"]].values * [image.shape[0], image.shape[1]],
            name="Cells",
            size=max(image.shape) * 0.002,
            edge_color=colors,
            face_color=colors,
        )
        viewer.title = str(document.path)
        napari.run()


@click.command()
@click.option(
    "--input",
    type=Path,
    required=True,
    help="Input project file / directory of project files to generate excel for.",
)
@click.option("--output", type=Path, required=True, help="Output excel file.")
@click.option(
    "--ignore-single-hemisphere",
    is_flag=True,
    help="Ignore slices that have only a single annotated hemisphere.",
)
@click.option(
    "--min-region-area-um2",
    type=int,
    default=250,
    show_default=True,
    help="Ignore regions that have area below this value (um^2).",
)
@click.option(
    "--cells-per-area-um2",
    type=int,
    default=250,
    show_default=True,
    help="Ignore regions that have area below this value (um^2).",
)
@click.option("--display", is_flag=True, help="Display cells on atlas.")
def create_excel_colabelling(
    input: Path,
    output: Path,
    ignore_single_hemisphere: bool,  # TODO: need to incorporate this more cleanly
    min_region_area_um2: int,
    cells_per_area_um2: int,
    display: bool,
):
    if (input / "brainways.bin").exists():
        paths = [input]
    else:
        paths = sorted(list(input.glob("*")))
    project = None
    cells_per_area_sheet = []
    cells_count_sheet = []
    for project_path in tqdm(paths):
        if project is None:
            project = BrainwaysProject.open(project_path)
            project.load_pipeline()
        else:
            project = BrainwaysProject.open(
                project_path, atlas=project.atlas, pipeline=project.pipeline
            )
            if project.settings.atlas != project.atlas.brainglobe_atlas.atlas_name:
                raise RuntimeError(
                    f"Multiple atlases detected: {project.settings.atlas},"
                    f" {project.atlas.brainglobe_atlas.atlas_name}"
                )

        cells_count_sheet.append(
            project.cell_count_summary_co_labeling(
                ignore_single_hemisphere=ignore_single_hemisphere,
                min_region_area_um2=min_region_area_um2,
            )
        )

        cells_per_area_sheet.append(
            project.cell_count_summary_co_labeling(
                ignore_single_hemisphere=ignore_single_hemisphere,
                min_region_area_um2=min_region_area_um2,
                cells_per_area_um2=cells_per_area_um2,
            )
        )

        if display:
            display_cells_3d(project)
            display_cells_2d(project)

    cells_count_sheet = pd.concat(
        [sheet for sheet in cells_count_sheet if sheet is not None], axis=0
    )
    cells_per_area_sheet = pd.concat(
        [sheet for sheet in cells_per_area_sheet if sheet is not None], axis=0
    )
    with ExcelWriter(output) as writer:
        cells_per_area_sheet.to_excel(
            writer, sheet_name=f"Cells per {cells_per_area_um2}um2", index=False
        )
        cells_count_sheet.to_excel(writer, sheet_name="Cell count", index=False)
