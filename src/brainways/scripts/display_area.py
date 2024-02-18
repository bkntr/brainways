import math
import random
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import scipy.ndimage
from skimage.util import view_as_blocks
from tqdm import tqdm

from brainways.pipeline.brainways_pipeline import PipelineStep
from brainways.project.brainways_project import BrainwaysProject
from brainways.utils._imports import NAPARI_AVAILABLE
from brainways.utils.cells import filter_cells_by_size


def update_layer_contrast_limits(
    layer,
    contrast_limits_quantiles: Tuple[float, float] = (0.01, 0.98),
    contrast_limits_range_quantiles: Tuple[float, float] = (0.0, 1.0),
) -> None:
    nonzero_mask = layer.data > 0
    if (~nonzero_mask).all():
        return

    limit_0, limit_1, limit_range_0, limit_range_1 = np.quantile(
        layer.data[nonzero_mask],
        (*contrast_limits_quantiles, *contrast_limits_range_quantiles),
    )
    layer.contrast_limits = (limit_0, limit_1 + 1e-8)
    layer.contrast_limits_range = (limit_range_0, limit_range_1 + 1e-8)


def gallery(im_in, ncols=None):
    n, h, w, c = im_in.shape
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n)))
    dn = (-n) % ncols  # trailing images
    im_out = np.empty((n + dn) * h * w * c, im_in.dtype).reshape(-1, w * ncols, c)
    view = view_as_blocks(im_out, (h, w, c))
    for k, im in enumerate(list(im_in) + dn * [0]):
        view[k // ncols, k % ncols, 0] = im
    return im_out


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
modes = ["struct_highres", "struct_lowres", "all_lowres", "atlas"]
mode = "struct_highres"


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

    project = BrainwaysProject.open(input)
    struct_id = project.atlas.brainglobe_atlas.structures[struct]["id"]
    all_struct_images = {}
    all_struct_annotations = {}
    all_struct_cells = {}
    all_lowres_images = {}
    all_lowres_full_annotations = {}
    all_atlas_slices = {}
    all_lowres_struct_annotations = {}
    all_lowres_annotations = {}
    all_subject_names = []
    subjects = random.sample(project.subjects, 1)
    for subject in tqdm(subjects):
        struct_images = []
        struct_annotations = []
        struct_cells = []
        for i, document in tqdm(subject.valid_documents):
            image_to_atlas_transform = project.pipeline.get_image_to_atlas_transform(
                document.params, document.lowres_image_size, until_step=PipelineStep.TPS
            )
            atlas_slice = subject.pipeline.atlas_registration.get_atlas_slice(
                document.params.atlas
            )
            annotation = atlas_slice.annotation.numpy()

            if struct_id not in annotation:
                continue

            annotation_on_atlas = atlas_slice.annotation.numpy()
            reference_on_atlas = atlas_slice.reference.numpy()

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

            highres_to_lowres_ratio = (
                document.image_size[0] / document.lowres_image_size[0]
            )
            annotation_on_highres_image = scipy.ndimage.zoom(
                annotation_on_lowres_image,
                zoom=highres_to_lowres_ratio,
                order=0,
            )
            cells = subject.get_valid_cells(document)
            cells = filter_cells_by_size(cells, min_size_um=25, max_size_um=125)
            cells_on_highres_image = cells[["x", "y"]].values * (
                document.image_size[1],
                document.image_size[0],
            )

            ys, xs = np.where(annotation_on_highres_image == struct_id)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            highres_image = (
                document.image_reader()
                .get_image_dask_data(
                    "YX",
                    X=slice(x0, x1),
                    Y=slice(y0, y1),
                    C=project.settings.channel,
                )
                .compute()
            )
            struct_images.append(highres_image)
            current_struct_annotation = (
                annotation_on_highres_image[y0:y1, x0:x1] == struct_id
            ).astype(np.uint8)
            struct_annotations.append(current_struct_annotation)
            struct_cells.append(
                cells_on_highres_image[
                    (cells_on_highres_image[:, 0] >= x0)
                    & (cells_on_highres_image[:, 0] <= x1)
                    & (cells_on_highres_image[:, 1] >= y0)
                    & (cells_on_highres_image[:, 1] <= y1)
                ]
                - (x0, y0)
            )

        if len(struct_images) == 0:
            continue

        struct_images = stack_pad_images(struct_images)
        struct_annotations = stack_pad_images(struct_annotations)
        struct_cells_disp = []
        for i, cells in enumerate(struct_cells):
            struct_cells_disp.append(
                np.concatenate([np.ones((len(cells), 1)) * i, cells[:, ::-1]], axis=1)
            )
        struct_cells_disp = np.concatenate(struct_cells_disp)

        subject_name = subject.subject_info.name
        all_struct_images[subject_name] = struct_images
        all_struct_annotations[subject_name] = struct_annotations
        all_struct_cells[subject_name] = struct_cells_disp
        all_lowres_struct_annotations[subject_name] = (
            annotation_on_lowres_image == struct_id
        ).astype(np.uint8)
        all_lowres_annotations[subject_name] = annotation_on_lowres_image
        all_lowres_full_annotations[subject_name] = annotation_on_atlas
        all_atlas_slices[subject_name] = reference_on_atlas
        all_lowres_images[subject_name] = lowres_image
        all_subject_names.append(subject_name)

    viewer = napari.Viewer()

    def skip_subject(skip: int):
        global cur_index
        viewer.layers.clear()
        cur_index = (cur_index + skip) % len(all_subject_names)
        subject_name = all_subject_names[cur_index]
        viewer.title = str(subject_name)

        # ["struct_highres", "struct_lowres", "all_lowres", "atlas"]
        if mode == "struct_highres":
            image_layer = viewer.add_image(
                all_struct_images[subject_name], name="highres image"
            )
            update_layer_contrast_limits(image_layer)
            viewer.add_labels(
                all_struct_annotations[subject_name], name="highres struct annotation"
            )
            viewer.add_points(
                all_struct_cells[subject_name], face_color="red", edge_color="red"
            )
        elif mode == "struct_lowres":
            lowres_image_layer = viewer.add_image(
                all_lowres_images[subject_name], name="lowres image"
            )
            update_layer_contrast_limits(lowres_image_layer)
            viewer.add_labels(
                all_lowres_struct_annotations[subject_name],
                name="lowres struct annotation",
            )
        elif mode == "all_lowres":
            lowres_image_layer = viewer.add_image(
                all_lowres_images[subject_name], name="lowres image"
            )
            update_layer_contrast_limits(lowres_image_layer)
            viewer.add_labels(
                all_lowres_annotations[subject_name],
                name="lowres struct annotation",
            )
        elif mode == "atlas":
            layer = viewer.add_image(all_atlas_slices[subject_name], name="atlas slice")
            update_layer_contrast_limits(layer)
            viewer.add_labels(
                all_lowres_full_annotations[subject_name], name="annotation"
            )

    def skip_mode(skip: int):
        global mode
        mode_index = (modes.index(mode) + skip) % len(modes)
        mode = modes[mode_index]
        skip_subject(0)

    @viewer.bind_key("n")
    def next(_):
        skip_subject(1)

    @viewer.bind_key("p")
    def prev(_):
        skip_subject(-1)

    @viewer.bind_key("l")
    def next_mode(_):
        skip_mode(1)

    @viewer.bind_key("j")
    def prev_mode(_):
        skip_mode(-1)

    skip_subject(1)

    napari.run()


if __name__ == "__main__":
    display_area()
