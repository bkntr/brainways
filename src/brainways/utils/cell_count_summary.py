import math
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cells import get_cell_struct_ids, get_parent_struct_ids


def set_co_labelling_product(cells: pd.DataFrame):
    cells = cells.copy()
    label_columns = [c for c in cells.columns if c.startswith("LABEL-")]
    if len(label_columns) == 0:
        return cells
    colabel_title_suffixes = ["neg", "pos"]
    for mask in product((False, True), repeat=len(label_columns)):
        colabel_subtitles = [
            f"{label_columns[i][len('LABEL-'):]}_{colabel_title_suffixes[mask[i]]}"
            for i in range(len(label_columns))
        ]
        colabel_title = "COLABEL-" + "-".join(colabel_subtitles)
        colabel_value = np.all(
            [cells[label_columns[i]] == mask[i] for i in range(len(label_columns))],
            axis=0,
        )
        cells.loc[:, colabel_title] = colabel_value
    return cells


def extend_cell_counts_to_parent_regions(
    cell_counts: pd.DataFrame,
    atlas: BrainwaysAtlas,
    structure_ids: Optional[List[int]] = None,
):
    if structure_ids is None:
        structure_ids = list(cell_counts.index)
    for struct_id in structure_ids:
        for parent_struct_id in get_parent_struct_ids(struct_id, atlas):
            if struct_id not in cell_counts.index:
                cell_counts.loc[struct_id] = 0
            if parent_struct_id not in cell_counts.index:
                cell_counts.loc[parent_struct_id] = cell_counts.loc[struct_id]
            else:
                cell_counts.loc[parent_struct_id] += cell_counts.loc[struct_id]

    return cell_counts


def extend_region_areas_to_parent_regions(
    region_areas: Dict[int, int],
    atlas: BrainwaysAtlas,
    structure_ids: Optional[List[int]] = None,
):
    if structure_ids is None:
        structure_ids = list(region_areas.keys())

    for struct_id in structure_ids:
        for parent_struct_id in get_parent_struct_ids(struct_id, atlas):
            if struct_id not in region_areas.keys():
                region_areas[struct_id] = 0
            if parent_struct_id not in region_areas:
                region_areas[parent_struct_id] = region_areas[struct_id]
            else:
                region_areas[parent_struct_id] += region_areas[struct_id]

    return region_areas


def get_cell_counts(cells: pd.DataFrame) -> pd.DataFrame:
    cells_grouped = cells.groupby("struct_id")
    cell_counts = cells_grouped.sum()
    label_columns = [
        c
        for c in cell_counts.columns
        if c.startswith("LABEL-") or c.startswith("COLABEL-")
    ]
    cell_counts_total = cells.groupby("struct_id")["x"].count()
    cell_counts = cell_counts[label_columns]
    cell_counts.loc[:, "cells"] = cell_counts_total
    return cell_counts


def get_struct_is_gray_matter(struct_id: int, atlas: BrainwaysAtlas) -> Optional[bool]:
    # TODO: this is atlas-specific
    if "GM" in atlas.brainglobe_atlas.structures.acronym_to_id_map:
        gray_matter_struct_id = atlas.brainglobe_atlas.structures.acronym_to_id_map[
            "GM"
        ]
        is_gray_matter = atlas.brainglobe_atlas.structures.tree.is_ancestor(
            gray_matter_struct_id, struct_id
        )
        return is_gray_matter
    else:
        return None


def cell_count_summary_co_labelling(
    animal_id: str,
    cells: pd.DataFrame,
    region_areas_um: Dict[int, int],
    atlas: BrainwaysAtlas,
    min_region_area_um2: Optional[int] = None,
    cells_per_area_um2: Optional[int] = None,
):
    cells = cells.copy()
    cells.loc[:, "struct_id"] = get_cell_struct_ids(
        cells=cells, bg_atlas=atlas.brainglobe_atlas
    )
    cells = set_co_labelling_product(cells)
    cell_counts = get_cell_counts(cells)
    all_leaf_structures = list(
        set(list(cell_counts.index) + list(region_areas_um.keys()))
    )
    cell_counts = extend_cell_counts_to_parent_regions(
        cell_counts=cell_counts, atlas=atlas, structure_ids=all_leaf_structures
    )
    region_areas_um = extend_region_areas_to_parent_regions(
        region_areas=region_areas_um, atlas=atlas, structure_ids=all_leaf_structures
    )

    if cells_per_area_um2:
        region_areas_um_list = [region_areas_um[i] for i in cell_counts.index]
        cell_counts = (
            cell_counts.div(region_areas_um_list, axis=0) * cells_per_area_um2**2
        )

    df = []
    atlas_structure_leave_ids = [
        node.identifier for node in atlas.brainglobe_atlas.structures.tree.leaves()
    ]

    for struct_id in region_areas_um:
        if struct_id not in atlas.brainglobe_atlas.structures:
            continue
        struct = atlas.brainglobe_atlas.structures[struct_id]

        if min_region_area_um2 is not None and region_areas_um[struct_id] < (
            min_region_area_um2**2
        ):
            continue

        df.append(
            {
                "animal_id": animal_id,
                "acronym": struct["acronym"],
                "name": struct["name"],
                "is_parent_structure": struct_id not in atlas_structure_leave_ids,
                "is_gray_matter": get_struct_is_gray_matter(
                    struct_id=struct_id, atlas=atlas
                ),
                "total_area_um2": int(math.sqrt(region_areas_um[struct_id])),
                **dict(cell_counts.loc[struct_id]),
            }
        )
    df = pd.DataFrame(df)
    return df
