from collections import Counter

import numpy as np
import pandas as pd
import pytest

from brainways.utils.cells import (
    cell_count_summary,
    cells_on_mask,
    filter_cells_on_mask,
    get_parent_struct_ids,
    get_region_areas,
)


def test_cell_counts(mock_atlas):
    cells = np.array([[0, 0, 0]])
    areas = Counter({10: 100})
    summary = cell_count_summary(cells=cells, region_areas=areas, atlas=mock_atlas)
    expected = pd.DataFrame(
        [
            {
                "id": 10,
                "acronym": "TEST",
                "name": "test_region",
                "cell_count": 1,
                "total_area_um2": 100,
                "cells_per_um2": 1 / 100,
            },
            {
                "id": 1,
                "acronym": "root",
                "name": "root",
                "cell_count": 1,
                "total_area_um2": 100,
                "cells_per_um2": 1 / 100,
            },
        ]
    )
    pd.testing.assert_frame_equal(summary, expected)


def test_cell_counts_min_region_area_um2(mock_atlas):
    cells = np.array([[0, 0, 0]])
    areas = Counter({10: 100})
    summary = cell_count_summary(
        cells=cells,
        region_areas=areas,
        atlas=mock_atlas,
        min_region_area_um2=200,
    )
    assert len(summary) == 0


def test_cell_counts_empty_region(mock_atlas):
    cells = np.array([[0, 0, 0]])
    areas = Counter({10: 100, 11: 100})
    summary = cell_count_summary(
        cells=cells,
        region_areas=areas,
        atlas=mock_atlas,
    )
    empty_region_cell_count = summary.loc[summary["id"] == 11, "cell_count"].item()
    assert empty_region_cell_count == 0


def test_region_areas(mock_atlas):
    registered_image = np.array([[0, 1], [1, 1]])
    annotation = np.array([[1, 1], [2, 3]])
    expected = {0: 6, 1: 1 * 6, 2: 1 * 6, 3: 1 * 6}
    result = get_region_areas(
        annotation=annotation,
        atlas=mock_atlas,
        registered_image=registered_image,
    )
    assert result == expected


def test_region_areas_mask(mock_atlas):
    registered_image = np.array([[1, 1], [1, 0]])
    annotation = np.array([[1, 1], [2, 3]])
    expected = {1: 2 * 6, 2: 1 * 6, 0: 1 * 6}
    result = get_region_areas(
        annotation=annotation,
        atlas=mock_atlas,
        registered_image=registered_image,
    )
    assert result == expected


def test_get_parent_struct_ids(mock_atlas):
    result = get_parent_struct_ids(struct_id=10, atlas=mock_atlas)
    expected = [1]
    assert result == expected


def test_filter_cells_on_tissue():
    cells = np.random.rand(50, 2) * (10, 10)
    tissue_mask = np.zeros((10, 10), dtype=bool)
    tissue_mask[:, 5:] = True
    filtered_cells = filter_cells_on_mask(cells=cells, mask=tissue_mask)
    expected = cells[cells[:, 0] > 5]
    assert (filtered_cells == expected).all()


def test_cells_on_mask_sets_outliers_to_false():
    cells = np.array([[10, 4]])
    mask = np.ones((20, 5), dtype=bool)
    cells_mask = cells_on_mask(cells=cells, mask=mask, ignore_outliers=True)
    assert np.all(cells_mask == [False])


def test_cells_on_mask_raises_error_on_outliers_when_no_ignore():
    cells = np.array([[10, 4]])
    mask = np.ones((20, 5), dtype=bool)
    with pytest.raises(IndexError):
        cells_on_mask(cells=cells, mask=mask, ignore_outliers=False)


def test_cells_on_mask_ignore_outliers_empty_cells():
    cells = np.empty((0, 2))
    mask = np.ones((20, 5), dtype=bool)
    filter_cells_on_mask(cells=cells, mask=mask, ignore_outliers=True)
