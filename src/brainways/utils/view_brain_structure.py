import pickle
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.ndimage
import tifffile
from tqdm import tqdm

from brainways.pipeline.brainways_pipeline import PipelineStep
from brainways.utils.io_utils.image_path import ImagePath

if TYPE_CHECKING:
    from brainways.project.brainways_project import BrainwaysProject


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


def create_image_grid(df: pd.DataFrame):
    positions = []
    next_x = 0
    next_y = 0
    row_height = 0
    grid_height = 0
    grid_width = 0
    prev_subject = None
    for _, row in df.iterrows():
        if row["subject"] != prev_subject:
            next_x = 0
            next_y += row_height
        positions.append((next_x, next_y))
        row_height = max(row_height, row["height"])
        next_x += row["width"]
        prev_subject = row["subject"]
        grid_width = max(next_x, grid_width)
        grid_height = next_y + row_height

    grid = np.zeros((grid_height, grid_width), dtype=df.iloc[0]["image"].dtype)
    for image, position in zip(df["image"], positions):
        x, y = position
        image_h, image_w = image.shape[0], image.shape[1]
        grid[y : y + image_h, x : x + image_w] = image
    return grid, positions


def create_cells_grid(cells_list: List[np.ndarray], positions: List[Tuple[int, int]]):
    grid = [cells + position for cells, position in zip(cells_list, positions)]
    grid = np.concatenate(grid)
    return grid


def view_brain_structure(
    project: "BrainwaysProject",
    structure_names: List[str],
    condition_type: Optional[str] = None,
    condition_value: Optional[str] = None,
    num_subjects: Optional[int] = None,
    display_channel: Optional[int] = None,
    filter_cell_type: Optional[str] = None,
):
    import napari

    condition_display_str = None
    if condition_type:
        assert condition_value
        condition_display_str = f"{condition_type}:{condition_value}"
    output_path = (
        project.path.parent
        / "structure_images"
        / f"{condition_type}_{condition_value}"
        / ",".join(structure_names)
    )
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.pkl"

    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        data = metadata["images_metadata"]
        for index, image_path in enumerate(sorted(output_path.glob("*.tiff"))):
            data[index]["image"] = tifffile.imread(image_path)
    else:
        data = load_data_from_project(
            project=project,
            structure_names=structure_names,
            output_path=output_path,
            condition_type=condition_type,
            condition_value=condition_value,
            num_subjects=num_subjects,
            display_channel=display_channel,
            filter_cell_type=filter_cell_type,
        )
        images_metadata = [
            {key: value for key, value in row.items() if key != "image"} for row in data
        ]
        metadata = {
            "images_metadata": images_metadata,
            "struct": ",".join(structure_names),
        }
        if condition_type is not None:
            metadata["condition"] = condition_display_str
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    data = pd.DataFrame(data).sort_values(by=["subject", "ap"])
    grid, positions = create_image_grid(data)
    cells_grid = create_cells_grid(data["cells"], positions=positions)

    viewer = napari.Viewer()
    layer = viewer.add_image(grid, colormap="green")
    update_layer_contrast_limits(layer)
    viewer.add_points(
        cells_grid[:, ::-1], face_color="red", border_color="red", size=50
    )

    image_shapes = [
        (row["image"].shape[0], row["image"].shape[1]) for _, row in data.iterrows()
    ]
    rectangles = np.array(
        [((y, x), (y + h, x + w)) for (x, y), (h, w) in zip(positions, image_shapes)]
    )
    viewer.add_shapes(
        rectangles,
        features={
            "text": [f"{row['subject']}/{int(row['ap'])}" for _, row in data.iterrows()]
        },
        shape_type="rectangle",
        edge_color="transparent",
        face_color="transparent",
        text={
            "string": "{text}",
            "anchor": "upper_left",
            "translation": [0, 0],
            "size": 10,
            "color": "red",
        },
        name="text",
        opacity=1.0,
    )
    if condition_type:
        viewer.title = condition_display_str


def load_data_from_project(
    project: "BrainwaysProject",
    structure_names: List[str],
    output_path: Path,
    condition_type: Optional[str] = None,
    condition_value: Optional[str] = None,
    num_subjects: Optional[int] = None,
    display_channel: Optional[int] = None,
    filter_cell_type: Optional[str] = None,
):
    struct_ids = [
        project.atlas.brainglobe_atlas.structures[structure_name]["id"]
        for structure_name in structure_names
    ]
    data = []
    subjects = project.subjects
    if condition_type:
        assert condition_value
        subjects = [
            subject
            for subject in subjects
            if subject.subject_info.conditions[condition_type] == condition_value
        ]

    if num_subjects is not None:
        subjects = project.subjects[:num_subjects]

    for subject in tqdm(subjects):
        for _, document in tqdm(subject.valid_documents):
            image_to_atlas_transform = project.pipeline.get_image_to_atlas_transform(
                document.params, document.lowres_image_size, until_step=PipelineStep.TPS
            )
            atlas_slice = subject.pipeline.atlas_registration.get_atlas_slice(
                document.params.atlas
            )
            annotation = atlas_slice.annotation.numpy()

            if not np.in1d(annotation, struct_ids).any():
                continue

            annotation_on_lowres_image = (
                image_to_atlas_transform.inv()
                .transform_image(
                    annotation.astype(np.float32),
                    output_size=document.lowres_image_size,
                    mode="nearest",
                )
                .astype(int)
            )
            annotation_on_lowres_image = np.in1d(
                annotation_on_lowres_image.flat, struct_ids
            )
            annotation_on_lowres_image = annotation_on_lowres_image.reshape(
                document.lowres_image_size
            )

            annotation_ccs, _ = scipy.ndimage.label(annotation_on_lowres_image)
            cc_indices = np.unique(annotation_ccs)

            highres_to_lowres_ratio = (
                document.image_size[0] / document.lowres_image_size[0]
            )
            annotation_ccs_highres = scipy.ndimage.zoom(
                annotation_ccs,
                zoom=highres_to_lowres_ratio,
                order=0,
            )

            cells = subject.get_valid_cells(document)
            if filter_cell_type:
                cells = cells[cells[f"LABEL-{filter_cell_type}"]]
            cells_on_highres_image = cells[["x", "y"]].values * (
                document.image_size[1],
                document.image_size[0],
            )

            for cc_index in cc_indices[1:]:
                ys, xs = np.where(annotation_ccs == cc_index)
                if len(xs) == 0:
                    continue
                x0, x1 = (
                    int(xs.min() * highres_to_lowres_ratio),
                    int((xs.max() + 1) * highres_to_lowres_ratio),
                )
                y0, y1 = (
                    int(ys.min() * highres_to_lowres_ratio),
                    int((ys.max() + 1) * highres_to_lowres_ratio),
                )
                annotation_cc = annotation_ccs_highres[y0:y1, x0:x1].copy()
                annotation_cc[annotation_cc > 0] = 1

                output_image_path = ImagePath(
                    filename=Path(document.path.filename).name,
                    scene=document.path.scene,
                )
                highres_image_cc_cache_path = (
                    output_path / f"{len(data):04}_{output_image_path}.tiff"
                )
                if highres_image_cc_cache_path.exists():
                    highres_image_cc = tifffile.imread(highres_image_cc_cache_path)
                else:
                    if display_channel is None:
                        display_channel = project.settings.channel
                    highres_image_cc = (
                        document.image_reader()
                        .get_image_dask_data(
                            "YX",
                            X=slice(x0, x1),
                            Y=slice(y0, y1),
                            C=display_channel,
                        )
                        .compute()
                    )
                    highres_image_cc[annotation_cc == 0] *= 0.8
                    tifffile.imwrite(highres_image_cc_cache_path, highres_image_cc)

                cells_cc = cells_on_highres_image[
                    (cells_on_highres_image[:, 0] >= x0)
                    & (cells_on_highres_image[:, 0] <= x1)
                    & (cells_on_highres_image[:, 1] >= y0)
                    & (cells_on_highres_image[:, 1] <= y1)
                ] - (x0, y0)
                # cells_cc = cells_cc[
                #     cells_on_mask(cells_cc, mask=annotation_cc, ignore_outliers=True)
                # ]

                data.append(
                    {
                        "image": highres_image_cc,
                        "height": highres_image_cc.shape[0],
                        "width": highres_image_cc.shape[1],
                        "cells": cells_cc,
                        "subject": subject.subject_info.name,
                        "ap": document.params.atlas.ap,
                    }
                )

        if len(data) == 0:
            continue
    return data
