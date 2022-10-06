import pandas as pd

from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cell_count_summary import (
    extend_cell_counts_to_parent_regions,
    get_cell_counts,
    set_co_labelling_product,
)


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
            "COLABEL-a_neg-b_neg": [True, False, False, False],
            "COLABEL-a_neg-b_pos": [False, True, False, False],
            "COLABEL-a_pos-b_neg": [False, False, True, False],
            "COLABEL-a_pos-b_pos": [False, False, False, True],
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
            "struct_id": [10, 1],
            "LABEL-a": [1, 1],
            "COLABEL-a_neg": [1, 1],
        }
    ).set_index("struct_id")

    region_areas = {10: 1}
    region_areas_expected = {10: 1, 1: 1}

    cell_counts_result, region_areas_result = extend_cell_counts_to_parent_regions(
        cell_counts=cell_counts, region_areas=region_areas, atlas=mock_atlas
    )
    pd.testing.assert_frame_equal(cell_counts_result, cell_counts_expected)
    assert region_areas_result == region_areas_expected


def test_get_cell_counts():
    cells = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5],
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
            "LABEL-a": [0, 2],
            "LABEL-b": [1, 1],
            "COLABEL-a_neg-b_neg": [1, 0],
            "COLABEL-a_neg-b_pos": [1, 0],
            "COLABEL-a_pos-b_neg": [0, 1],
            "COLABEL-a_pos-b_pos": [0, 1],
            "cells": [2, 2],
        }
    ).set_index("struct_id")
    result = get_cell_counts(cells)
    pd.testing.assert_frame_equal(result, expected)
