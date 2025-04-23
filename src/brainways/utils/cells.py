import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
from skimage.measure import regionprops_table

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline, PipelineStep
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


def get_cell_struct_ids(cells: pd.DataFrame, bg_atlas: BrainGlobeAtlas) -> np.ndarray:
    struct_ids = []
    cells_xy = cells[["x", "y", "z"]].values
    for cell in cells_xy:
        try:
            struct_id = bg_atlas.structure_from_coords(
                cell[::-1].round().astype(int).tolist()
            )
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


def get_region_areas(
    annotation: np.ndarray, atlas: BrainwaysAtlas, mask: np.ndarray
) -> Dict[int, int]:
    """
    area in μm^2
    :param annotation:
    :param atlas:
    :return: {struct_id: area (μm^2)}
    """
    masked_annotation = annotation * mask
    pixel_to_um2 = (
        atlas.brainglobe_atlas.resolution[1] * atlas.brainglobe_atlas.resolution[2]
    )
    struct_ids, areas_pixel = np.unique(masked_annotation, return_counts=True)
    region_areas_um2 = Counter()
    for struct_id, area in zip(struct_ids.tolist(), areas_pixel.tolist()):
        region_areas_um2[struct_id] += area * pixel_to_um2
    return region_areas_um2


def get_parent_struct_ids(struct_id: int, atlas: BrainwaysAtlas) -> List[int]:
    if struct_id not in atlas.brainglobe_atlas.structures:
        return []

    parents = []
    while True:
        parent = atlas.brainglobe_atlas.structures.tree.parent(struct_id)
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
    cells: pd.DataFrame, mask: np.ndarray, ignore_outliers: bool = False
) -> pd.DataFrame:
    cells_np = cells.loc[:, ["x", "y"]].values
    filtered_cells = cells.loc[
        cells_on_mask(cells_np, mask, ignore_outliers=ignore_outliers)
    ]
    return filtered_cells.copy()


def filter_cells_on_tissue(cells: pd.DataFrame, image: np.ndarray) -> pd.DataFrame:
    cells.loc[:, "x"] *= image.shape[1]
    cells.loc[:, "y"] *= image.shape[0]
    tissue_mask = brain_mask_simple(image)
    cells = filter_cells_on_mask(cells=cells, mask=tissue_mask)
    cells.loc[:, "x"] /= image.shape[1]
    cells.loc[:, "y"] /= image.shape[0]
    return cells


def filter_cells_on_annotation(
    cells: pd.DataFrame,
    lowres_image_size: ImageSizeHW,
    params: BrainwaysParams,
    pipeline: BrainwaysPipeline,
    annotation: np.ndarray,
) -> pd.DataFrame:
    cells_on_image = cells[["x", "y"]].values * lowres_image_size[::-1]
    image_to_atlas_transform = pipeline.get_image_to_atlas_transform(
        brainways_params=params,
        lowres_image_size=lowres_image_size,
        until_step=PipelineStep.TPS,
    )
    cells_on_slice = image_to_atlas_transform.transform_points(cells_on_image)
    cells_on_slice_mask = cells_on_mask(
        cells=cells_on_slice, mask=annotation > 0, ignore_outliers=True
    )
    result = cells.loc[cells_on_slice_mask]
    return result.copy()


def filter_cells_by_size(
    cells: pd.DataFrame,
    min_size_um: Optional[float] = None,
    max_size_um: Optional[float] = None,
):
    if not min_size_um and not max_size_um:
        return cells

    if "area_um" not in cells.columns and "area_pixels" not in cells.columns:
        logging.warning("Cells do not have size information, skipping size filtering.")
        return cells

    query = np.ones(len(cells), dtype=bool)
    if cells["area_um"].isna().any():
        logging.warning(
            "Images do not have pixel size information, filtering cell size by pixels."
        )
        cell_sizes = cells["area_pixels"]
    else:
        cell_sizes = cells["area_um"]

    if min_size_um:
        query &= cell_sizes >= min_size_um
    if max_size_um:
        query &= cell_sizes <= max_size_um
    return cells[query]
