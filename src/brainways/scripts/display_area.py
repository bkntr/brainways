from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import scipy.ndimage
from tqdm import tqdm

from brainways.project.brainways_subject import BrainwaysSubject
from brainways.utils._imports import NAPARI_AVAILABLE


def stack_pad_images(images: List[np.ndarray]) -> np.ndarray:
    stack_height = max(image.shape[0] for image in images)
    stack_width = max(image.shape[1] for image in images)
    stack_image = np.zeros(
        (len(images), stack_height, stack_width), dtype=images[0].dtype
    )
    for i, image in enumerate(images):
        stack_image[i, : image.shape[0], : image.shape[1]] = image
    return stack_image


cur_index = -1


@click.command()
@click.option(
    "--input",
    type=Path,
    help="Input directory of subjects to display area for.",
)
@click.option("--struct", help="Structure acronym to display")
@click.option("--print-all", is_flag=True, help="Print all structures to console.")
def display_area(input: Path, struct: str, print_all: bool):
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Please install napari to display area: "
            "`pip install napari` or `pip install brainways[all]`"
        ) from None

    import napari

    paths = list(input.glob("*"))
    all_struct_images = {}
    all_struct_annotations = {}
    all_struct_cells = {}
    subject: Optional[BrainwaysSubject] = None
    for subject_path in tqdm(paths):
        if subject is not None:
            subject = BrainwaysSubject.open(
                subject_path, atlas=subject.atlas, pipeline=subject.pipeline
            )
        else:
            subject = BrainwaysSubject.open(subject_path)
            subject.load_pipeline()
        struct_id = subject.pipeline.atlas.brainglobe_atlas.structures[struct]["id"]
        struct_images = []
        struct_annotations = []
        struct_cells = []
        for i, document in tqdm(subject.valid_documents):
            image_to_atlas_transform = subject.pipeline.get_image_to_atlas_transform(
                document.params, document.lowres_image_size
            )
            atlas_slice = subject.pipeline.atlas_registration.get_atlas_slice(
                document.params.atlas
            )
            annotation = atlas_slice.annotation.numpy()

            if struct_id not in annotation:
                continue

            lowres_image = subject.read_lowres_image(document)
            annotation_on_lowres_image = (
                image_to_atlas_transform.inv()
                .transform_image(
                    annotation.astype(np.float32),
                    output_size=lowres_image.shape,
                    mode="nearest",
                )
                .astype(int)
            )

            highres_image = subject.read_highres_image(document)
            highres_to_lowres_ratio = highres_image.shape[0] / lowres_image.shape[0]
            annotation_on_highres_image = scipy.ndimage.zoom(
                annotation_on_lowres_image,
                zoom=highres_to_lowres_ratio,
                order=0,
            )
            cells = subject.get_valid_cells(document)
            cells_on_highres_image = cells[["x", "y"]].values * (
                highres_image.shape[1],
                highres_image.shape[0],
            )

            ys, xs = np.where(annotation_on_highres_image == struct_id)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            struct_images.append(highres_image[y0:y1, x0:x1])
            struct_annotations.append(annotation_on_highres_image[y0:y1, x0:x1])
            struct_cells.append(
                cells_on_highres_image[
                    (cells_on_highres_image[:, 0] >= x0)
                    & (cells_on_highres_image[:, 0] <= x1)
                    & (cells_on_highres_image[:, 1] >= y0)
                    & (cells_on_highres_image[:, 1] <= y1)
                ]
                - (x0, y0)
            )

        struct_images = stack_pad_images(struct_images)
        struct_annotations = stack_pad_images(struct_annotations)
        struct_cells_disp = []
        for i, cells in enumerate(struct_cells):
            struct_cells_disp.append(
                np.concatenate([np.ones((len(cells), 1)) * i, cells[:, ::-1]], axis=1)
            )
        struct_cells_disp = np.concatenate(struct_cells_disp)

        all_struct_images[subject_path] = struct_images
        all_struct_annotations[subject_path] = struct_annotations
        all_struct_cells[subject_path] = struct_cells_disp
        break

    viewer = napari.Viewer()

    def next_prev(next: bool):
        global cur_index
        viewer.layers.clear()
        if next:
            cur_index = min(cur_index + 1, len(paths))
        else:
            cur_index = max(cur_index - 1, 0)
        path = paths[cur_index]
        viewer.title = str(path)
        viewer.add_image(all_struct_images[path])
        viewer.add_labels(all_struct_annotations[path])
        viewer.add_points(all_struct_cells[path], face_color="red", edge_color="red")

    @viewer.bind_key("n")
    def next(_):
        next_prev(True)

    @viewer.bind_key("p")
    def prev(_):
        next_prev(False)

    next_prev(True)

    napari.run()
