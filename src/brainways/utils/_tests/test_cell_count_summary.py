import math

import pandas as pd

from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cell_count_summary import (
    cell_count_summary,
    extend_cell_counts_to_parent_regions,
    extend_region_areas_to_parent_regions,
    get_cell_counts,
    set_co_labelling_product,
)
from brainways.utils.cells import get_cell_struct_ids


def test_set_co_labelling_product():
    cells = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "LABEL-a": [False, False, True, True],
            "LABEL-b": [False, True, False, True],
        }
    )

    expected = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "LABEL-a": [False, False, True, True],
            "LABEL-b": [False, True, False, True],
            "COLABEL-a-b-": [True, False, False, False],
            "COLABEL-a-b+": [False, True, False, False],
            "COLABEL-a+b-": [False, False, True, False],
            "COLABEL-a+b+": [False, False, False, True],
        }
    )

    result = set_co_labelling_product(cells)
    pd.testing.assert_frame_equal(result, expected)


def test_set_co_labelling_product_no_labels():
    cells = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
        }
    )

    expected = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
        }
    )

    result = set_co_labelling_product(cells)
    pd.testing.assert_frame_equal(result, expected)


def test_set_co_labelling_product_one_label():
    cells = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "LABEL-a": [True, True, False, False],
        }
    )

    expected = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "LABEL-a": [True, True, False, False],
        }
    )

    result = set_co_labelling_product(cells)
    pd.testing.assert_frame_equal(result, expected)


def test_extend_cell_counts_to_parent_regions(mock_atlas: BrainwaysAtlas):
    cell_counts = pd.DataFrame(
        {
            "struct_id": [10],
            "LABEL-a": [1],
            "COLABEL-a_neg": [1],
        }
    ).set_index("struct_id")
    cell_counts_expected = pd.DataFrame(
        {
            "struct_id": [10, 1, 11],
            "LABEL-a": [1, 1, 0],
            "COLABEL-a_neg": [1, 1, 0],
        }
    ).set_index("struct_id")

    region_areas = {10: 1, 11: 1}
    region_areas_expected = {10: 1, 1: 2, 11: 1}

    cell_counts_result = extend_cell_counts_to_parent_regions(
        cell_counts=cell_counts, atlas=mock_atlas, structure_ids=[10, 1, 11]
    )
    region_areas_result = extend_region_areas_to_parent_regions(
        region_areas=region_areas, atlas=mock_atlas, structure_ids=[10, 1, 11]
    )
    pd.testing.assert_frame_equal(cell_counts_result, cell_counts_expected)
    assert region_areas_result == region_areas_expected


def test_get_cell_counts():
    cells = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "z": [0.5, 0.5, 0.5, 0.5],
            "LABEL-a": [False, False, True, True],
            "LABEL-b": [False, True, False, True],
            "COLABEL-a_neg-b_neg": [True, False, False, False],
            "COLABEL-a_neg-b_pos": [False, True, False, False],
            "COLABEL-a_pos-b_neg": [False, False, True, False],
            "COLABEL-a_pos-b_pos": [False, False, False, True],
            "struct_id": [0, 0, 1, 1],
        }
    )
    expected = pd.DataFrame(
        {
            "struct_id": [0, 1],
            "cells": [2, 2],
            "LABEL-a": [0, 2],
            "LABEL-b": [1, 1],
            "COLABEL-a_neg-b_neg": [1, 0],
            "COLABEL-a_neg-b_pos": [1, 0],
            "COLABEL-a_pos-b_neg": [0, 1],
            "COLABEL-a_pos-b_pos": [0, 1],
        }
    ).set_index("struct_id")
    result = get_cell_counts(cells)
    pd.testing.assert_frame_equal(result, expected)


def test_get_cell_struct_ids(mock_atlas: BrainwaysAtlas):
    cells = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "z": [0.5, 0.5, 0.5, 0.5],
        }
    )
    struct_ids = get_cell_struct_ids(cells, mock_atlas.brainglobe_atlas)
    expected_struct_ids = [10, 10, 10, 10]
    assert all(struct_ids == expected_struct_ids)


def test_get_cell_count_summary_co_labeling(mock_atlas: BrainwaysAtlas):
    cells = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "z": [0.5, 0.5, 0.5, 0.5],
            "struct_id": [10, 10, 10, 10],
            "LABEL-a": [False, False, True, True],
            "LABEL-b": [False, True, False, True],
        }
    )
    region_areas = {10: 1}
    expected = pd.DataFrame(
        [
            {
                "animal_id": "test",
                "acronym": "TEST",
                "name": "test_region",
                "is_parent_structure": False,
                "is_gray_matter": None,
                "total_area_um2": 1,
                "cells": 4,
                "a+": 2,
                "b+": 2,
                "a-b-": 1,
                "a-b+": 1,
                "a+b-": 1,
                "a+b+": 1,
            },
            {
                "animal_id": "test",
                "acronym": "root",
                "name": "root",
                "is_parent_structure": True,
                "is_gray_matter": None,
                "total_area_um2": 1,
                "cells": 4,
                "a+": 2,
                "b+": 2,
                "a-b-": 1,
                "a-b+": 1,
                "a+b-": 1,
                "a+b+": 1,
            },
        ]
    )
    result = cell_count_summary(
        animal_id="test", cells=cells, region_areas_um=region_areas, atlas=mock_atlas
    )
    pd.testing.assert_frame_equal(result, expected)


def test_get_cell_count_summary_co_labeling_cells_per_area(mock_atlas: BrainwaysAtlas):
    cells = pd.DataFrame(
        {
            "struct_id": [10, 10, 10, 10],
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
            "z": [0.5, 0.5, 0.5, 0.5],
            "LABEL-a": [False, False, True, True],
            "LABEL-b": [False, True, False, True],
        }
    )
    region_areas_um = {10: 16}
    expected = pd.DataFrame(
        [
            {
                "animal_id": "test",
                "acronym": "TEST",
                "name": "test_region",
                "is_parent_structure": False,
                "is_gray_matter": None,
                "total_area_um2": 16,
                "cells": 8.0,
                "a+": 4.0,
                "b+": 4.0,
                "a-b-": 2.0,
                "a-b+": 2.0,
                "a+b-": 2.0,
                "a+b+": 2.0,
                "cells (not normalized)": 4.0,
                "a+ (not normalized)": 2.0,
                "b+ (not normalized)": 2.0,
                "a-b- (not normalized)": 1.0,
                "a-b+ (not normalized)": 1.0,
                "a+b- (not normalized)": 1.0,
                "a+b+ (not normalized)": 1.0,
            },
            {
                "animal_id": "test",
                "acronym": "root",
                "name": "root",
                "is_parent_structure": True,
                "is_gray_matter": None,
                "total_area_um2": 16,
                "cells": 8.0,
                "a+": 4.0,
                "b+": 4.0,
                "a-b-": 2.0,
                "a-b+": 2.0,
                "a+b-": 2.0,
                "a+b+": 2.0,
                "cells (not normalized)": 4.0,
                "a+ (not normalized)": 2.0,
                "b+ (not normalized)": 2.0,
                "a-b- (not normalized)": 1.0,
                "a-b+ (not normalized)": 1.0,
                "a+b- (not normalized)": 1.0,
                "a+b+ (not normalized)": 1.0,
            },
        ]
    )
    result = cell_count_summary(
        animal_id="test",
        cells=cells,
        region_areas_um=region_areas_um,
        atlas=mock_atlas,
        cells_per_area_um2=math.sqrt(32),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_get_cell_count_summary_conditions(mock_atlas: BrainwaysAtlas):
    cells = pd.DataFrame(
        {
            "x": [0.5],
            "y": [0.5],
            "z": [0.5],
            "struct_id": [10],
        }
    )
    region_areas = {10: 1}
    expected = pd.DataFrame(
        [
            {
                "condition1": "c1",
                "condition2": "c2",
                "animal_id": "test",
                "acronym": "TEST",
                "name": "test_region",
                "is_parent_structure": False,
                "is_gray_matter": None,
                "total_area_um2": 1,
                "cells": 1,
            },
            {
                "condition1": "c1",
                "condition2": "c2",
                "animal_id": "test",
                "acronym": "root",
                "name": "root",
                "is_parent_structure": True,
                "is_gray_matter": None,
                "total_area_um2": 1,
                "cells": 1,
            },
        ]
    )
    result = cell_count_summary(
        animal_id="test",
        cells=cells,
        region_areas_um=region_areas,
        atlas=mock_atlas,
        conditions={"condition1": "c1", "condition2": "c2"},
    )
    pd.testing.assert_frame_equal(result, expected)
