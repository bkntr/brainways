from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from bg_atlasapi import BrainGlobeAtlas
from skimage.measure import regionprops_table

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline, PipelineStep
from brainways.project.brainways_project_settings import ProjectDocument
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.image import ImageSizeHW, brain_mask_simple


def cell_mask_to_points(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    regionprops_df = pd.DataFrame(
        regionprops_table(
            mask, image, properties=("centroid", "area", "mean_intensity")
        )
    )

    return (
        regionprops_df[["centroid-1", "centroid-0"]].to_numpy().round().astype(np.int32)
    )


def get_cell_struct_ids(cells: np.ndarray, bg_atlas: BrainGlobeAtlas) -> np.ndarray:
    struct_ids = []
    for cell in cells:
        try:
            struct_id = bg_atlas.structure_from_coords(cell[::-1].astype(int).tolist())
        except IndexError:
            struct_id = 0
        struct_ids.append(struct_id)
    return np.array(struct_ids)


def get_struct_colors(struct_ids: np.ndarray, bg_atlas: BrainGlobeAtlas):
    colors = []
    for struct_id in struct_ids:
        if struct_id == 0:
            color = [0, 0, 0, 255]
        else:
            color = bg_atlas.structures[struct_id]["rgb_triplet"] + [255]
        colors.append(color)
    colors = np.array(colors) / 255
    return colors


def cell_count_summary(
    cells: np.ndarray,
    region_areas: Dict[int, int],
    atlas: BrainwaysAtlas,
    min_region_area_um2: Optional[int] = None,
):
    struct_ids = get_cell_struct_ids(cells=cells, bg_atlas=atlas.atlas)
    cell_counts = Counter()
    for struct_id in struct_ids.tolist():
        cell_counts[struct_id] += 1

    # extend cell counts and region areas to parent structures
    all_leaf_structures = set(list(cell_counts.keys()) + list(region_areas.keys()))
    for struct_id in all_leaf_structures:
        for parent_struct_id in get_parent_struct_ids(struct_id, atlas):
            cell_counts[parent_struct_id] += cell_counts[struct_id]
            region_areas[parent_struct_id] += region_areas[struct_id]

    df = []
    for struct_id in region_areas:
        if struct_id not in atlas.atlas.structures:
            continue
        struct = atlas.atlas.structures[struct_id]

        if (
            min_region_area_um2 is not None
            and region_areas[struct_id] < min_region_area_um2
        ):
            continue

        df.append(
            {
                "id": struct["id"],
                "acronym": struct["acronym"],
                "name": struct["name"],
                "cell_count": int(cell_counts[struct_id]),
                "total_area_um2": int(region_areas[struct_id]),
                "cells_per_um2": float(
                    cell_counts[struct_id] / max(region_areas[struct_id], 1)
                ),
            }
        )
    df = pd.DataFrame(df)
    return df


def get_region_areas(
    annotation: np.ndarray, atlas: BrainwaysAtlas, registered_image: np.ndarray
) -> Dict[int, int]:
    """
    area in μm^2
    :param annotation:
    :param atlas:
    :return: {struct_id: area (μm^2)}
    """
    mask = brain_mask_simple(registered_image)
    masked_annotation = annotation * mask
    pixel_to_um2 = atlas.atlas.resolution[1] * atlas.atlas.resolution[2]
    struct_ids, areas_pixel = np.unique(masked_annotation, return_counts=True)
    region_areas_um2 = Counter()
    for struct_id, area in zip(struct_ids.tolist(), areas_pixel.tolist()):
        region_areas_um2[struct_id] += area * pixel_to_um2
    return region_areas_um2


def get_parent_struct_ids(struct_id: int, atlas: BrainwaysAtlas) -> List[int]:
    if struct_id not in atlas.atlas.structures:
        return []

    parents = []
    while True:
        parent = atlas.atlas.structures.tree.parent(struct_id)
        if parent is None:
            break

        struct_id = parent.identifier
        parents.append(struct_id)
    return parents


def cells_on_mask(
    cells: np.ndarray, mask: np.ndarray, ignore_outliers: bool = False
) -> np.ndarray:
    outlier_cells = None

    if ignore_outliers:
        outlier_cells = (
            (cells[:, 0] < 0)
            | (cells[:, 0] >= mask.shape[1])
            | (cells[:, 1] < 0)
            | (cells[:, 1] >= mask.shape[0])
        )
        cells[outlier_cells] = (0, 0)

    cells_on_mask = mask[cells[:, 1].astype(int), cells[:, 0].astype(int)]

    if ignore_outliers:
        cells_on_mask[outlier_cells] = False

    return cells_on_mask


def filter_cells_on_mask(
    cells: np.ndarray,
    mask: np.ndarray,
    ignore_outliers: bool = False,
) -> np.ndarray:
    filtered_cells = cells[cells_on_mask(cells, mask, ignore_outliers=ignore_outliers)]
    return filtered_cells


def filter_cells_on_tissue(cells: np.ndarray, image: np.ndarray) -> np.ndarray:
    cells = cells * image.shape[::-1]
    tissue_mask = brain_mask_simple(image)
    cells = filter_cells_on_mask(cells=cells, mask=tissue_mask)
    cells = cells / image.shape[::-1]
    return cells


def filter_cells_on_annotation(
    cells: np.ndarray,
    lowres_image_size: ImageSizeHW,
    params: BrainwaysParams,
    pipeline: BrainwaysPipeline,
) -> np.ndarray:
    annotation = pipeline.get_atlas_slice(params).annotation.numpy()
    cells_on_image = cells * lowres_image_size[::-1]
    image_to_atlas_transform = pipeline.get_image_to_atlas_transform(
        brainways_params=params,
        lowres_image_size=lowres_image_size,
        until_step=PipelineStep.TPS,
    )
    cells_on_slice = image_to_atlas_transform.transform_points(cells_on_image)
    cells_on_slice_mask = cells_on_mask(
        cells=cells_on_slice, mask=annotation > 0, ignore_outliers=True
    )
    result = cells[cells_on_slice_mask]
    return result


def read_cells_csv(csv_path: Path, document: ProjectDocument):
    # TODO: refactor
    with open(csv_path) as f:
        header = f.readline()
    if header[:5] == "Image":
        aicsimage = AICSImage(document.path.filename)
        aicsimage.set_scene(document.path.scene)
        cells_df = pd.read_csv(csv_path, sep="\t")
        cfos_cells = cells_df["Name"].str.contains("cFos")
        cells = cells_df.loc[cfos_cells, ["Centroid X µm", "Centroid Y µm"]].values
        image_size_um = [
            aicsimage.dims.X * aicsimage.physical_pixel_sizes.X,
            aicsimage.dims.Y * aicsimage.physical_pixel_sizes.Y,
        ]
        cells = cells / image_size_um
    else:
        cells_df = pd.read_csv(csv_path)
        cells = cells_df[["centroid-1", "centroid-0"]].to_numpy()
        if (cells > 1).any():
            cells = cells / document.image_size[::-1]
    assert (cells < 1).all()
    return cells
