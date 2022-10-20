from pathlib import Path

import click
import pandas as pd
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
    struct_ids = get_cell_struct_ids(all_cells, project.atlas.atlas)
    colors = get_struct_colors(struct_ids, project.atlas.atlas)

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
        if document.cells is None:
            continue

        viewer = napari.Viewer()
        image = project.read_highres_image(document, level=1)
        viewer.add_image(image, name=str(document.path))

        cells_atlas = project.get_cells_on_atlas([document])
        cells = project.get_valid_cells(document)
        struct_ids = get_cell_struct_ids(cells_atlas, project.atlas.atlas)
        colors = get_struct_colors(struct_ids, project.atlas.atlas)

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
    "--min-region-area-um2",
    type=int,
    default=250 * 250,
    show_default=True,
    help="Ignore regions that have area below this value (um^2).",
)
@click.option("--display", is_flag=True, help="Display cells on atlas.")
def create_excel_colabelling(
    input: Path, output: Path, min_region_area_um2: int, display: bool
):
    if (input / "brainways.bin").exists():
        paths = [input]
    else:
        paths = sorted(list(input.glob("*")))
    project = None
    excel = []
    for project_path in tqdm(paths):
        if project is None:
            project = BrainwaysProject.open(project_path)
            project.load_pipeline()
        else:
            project = BrainwaysProject.open(
                project_path, atlas=project.atlas, pipeline=project.pipeline
            )
            if project.settings.atlas != project.atlas.atlas.atlas_name:
                raise RuntimeError(
                    f"Multiple atlases detected: {project.settings.atlas},"
                    f" {project.atlas.atlas.atlas_name}"
                )

        project_summary = project.cell_count_summary_co_labeling(
            min_region_area_um2=min_region_area_um2
        )
        if project_summary is None:
            continue
        project_summary.insert(0, "animal_id", project.project_path.stem)
        excel.append(project_summary)

        if display:
            display_cells_3d(project)
            display_cells_2d(project)

    excel = pd.concat(excel, axis=0)
    excel.to_excel(output, index=False)
