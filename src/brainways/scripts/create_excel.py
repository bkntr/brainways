import logging
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
    struct_ids = get_cell_struct_ids(all_cells, project.atlas.brainglobe_atlas)
    colors = get_struct_colors(struct_ids, project.atlas.brainglobe_atlas)

    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3
    viewer.add_image(project.atlas.reference.numpy(), name="Atlas")
    viewer.add_points(
        all_cells[:, ::-1],
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
        image = project.read_highres_image(document)
        viewer.add_image(image, name=str(document.path))

        cells_atlas = project.get_cells_on_atlas([document])
        cells = project.get_valid_cells(document)
        struct_ids = get_cell_struct_ids(cells_atlas, project.atlas.brainglobe_atlas)
        colors = get_struct_colors(struct_ids, project.atlas.brainglobe_atlas)

        viewer.add_points(
            cells[:, ::-1] * [image.shape[0], image.shape[1]],
            name="Cells",
            size=50,
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
def create_excel(input: Path, output: Path, min_region_area_um2: int, display: bool):
    if (input / "brainways.bin").exists():
        paths = [input]
    else:
        paths = sorted(list(input.glob("*")))
    project = None
    cell_count_sheet = []
    total_area_um2_sheet = []
    cells_per_250um2_sheet = []
    for project_path in tqdm(paths):
        if project is None:
            project = BrainwaysProject.open(project_path)
            project.load_pipeline()

            structures = {
                struct["acronym"]: struct["name"]
                for struct in project.atlas.brainglobe_atlas.structures.values()
            }
            first_row = {"animal": "", **structures}
            cell_count_sheet += [first_row]
            total_area_um2_sheet += [first_row]
            cells_per_250um2_sheet += [first_row]
        else:
            project = BrainwaysProject.open(
                project_path, atlas=project.atlas, pipeline=project.pipeline
            )
            if project.settings.atlas != project.atlas.brainglobe_atlas.atlas_name:
                raise RuntimeError(
                    f"Multiple atlases detected: {project.settings.atlas},"
                    f" {project.atlas.brainglobe_atlas.atlas_name}"
                )
        summary = project.cell_count_summary(min_region_area_um2=min_region_area_um2)
        cell_count_sheet.append(
            {
                "animal": project_path.stem,
                **dict(zip(summary["acronym"], summary["cell_count"])),
            }
        )
        total_area_um2_sheet.append(
            {
                "animal": project_path.stem,
                **dict(zip(summary["acronym"], summary["total_area_um2"])),
            }
        )
        cells_per_250um2_sheet.append(
            {
                "animal": project_path.stem,
                **dict(
                    zip(
                        summary["acronym"],
                        summary["cells_per_um2"] * (250 * 250),
                    )
                ),
            }
        )

        if display:
            display_cells_3d(project)
            display_cells_2d(project)

    cells_per_250um2_sheet = pd.DataFrame(cells_per_250um2_sheet)
    total_area_um2_sheet = pd.DataFrame(total_area_um2_sheet)
    cell_count_sheet = pd.DataFrame(cell_count_sheet)
    struct_leafs = [
        project.atlas.brainglobe_atlas.structures[node.identifier]["acronym"]
        for node in project.atlas.brainglobe_atlas.structures.tree.leaves()
    ]
    cells_per_250um2_leaves_sheet = cells_per_250um2_sheet[["animal"] + struct_leafs]
    total_area_um2_leaves_sheet = total_area_um2_sheet[["animal"] + struct_leafs]
    cell_count_leaves_sheet = cell_count_sheet[["animal"] + struct_leafs]
    with pd.ExcelWriter(output) as writer:
        cells_per_250um2_leaves_sheet.to_excel(
            writer, sheet_name="Cells per 250um2", index=False
        )
        total_area_um2_leaves_sheet.to_excel(
            writer, sheet_name="Total Area (um)", index=False
        )
        cell_count_leaves_sheet.to_excel(writer, sheet_name="Cell Count", index=False)

        cells_per_250um2_sheet.to_excel(
            writer, sheet_name="Cells per 250um2 (Full)", index=False
        )
        total_area_um2_sheet.to_excel(
            writer, sheet_name="Total Area (um) (Full)", index=False
        )
        cell_count_sheet.to_excel(writer, sheet_name="Cell Count (Full)", index=False)
