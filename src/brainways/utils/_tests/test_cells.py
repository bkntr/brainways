import numpy as np
import pandas as pd
import pytest

from brainways.utils.cells import (
    cells_on_mask,
    filter_cells_by_size,
    filter_cells_on_mask,
    get_parent_struct_ids,
    get_region_areas,
)


def test_region_areas(mock_atlas):
    registered_image = np.array([[0, 1], [1, 1]])
    annotation = np.array([[1, 1], [2, 3]])
    expected = {0: 600, 1: 1 * 600, 2: 1 * 600, 3: 1 * 600}
    result = get_region_areas(
        annotation=annotation,
        atlas=mock_atlas,
        mask=registered_image,
    )
    assert result == expected


def test_region_areas_mask(mock_atlas):
    registered_image = np.array([[1, 1], [1, 0]])
    annotation = np.array([[1, 1], [2, 3]])
    expected = {1: 2 * 600, 2: 1 * 600, 0: 1 * 600}
    result = get_region_areas(
        annotation=annotation,
        atlas=mock_atlas,
        mask=registered_image,
    )
    assert result == expected


def test_get_parent_struct_ids(mock_atlas):
    result = get_parent_struct_ids(struct_id=10, atlas=mock_atlas)
    expected = [1]
    assert result == expected


def test_filter_cells_on_tissue():
    cells = np.random.rand(50, 2) * (10, 10)
    cells = pd.DataFrame({"x": cells[:, 0], "y": cells[:, 1]})
    tissue_mask = np.zeros((10, 10), dtype=bool)
    tissue_mask[:, 5:] = True
    filtered_cells = filter_cells_on_mask(cells=cells, mask=tissue_mask)
    expected = cells[cells["x"] > 5]
    assert (filtered_cells[["x", "y"]].values == expected[["x", "y"]].values).all()


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
    cells = pd.DataFrame({"x": [0], "y": [2]})
    mask = np.ones((20, 5), dtype=bool)
    filter_cells_on_mask(cells=cells, mask=mask, ignore_outliers=True)


def test_filter_cells_by_size_with_min_and_max():
    cells = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "area_um": [1, 10, 15]})
    filtered_cells = filter_cells_by_size(cells=cells, min_size_um=5, max_size_um=10)
    expected = cells.loc[[1]]
    pd.testing.assert_frame_equal(filtered_cells, expected)


def test_filter_cells_by_size_with_min():
    cells = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "area_um": [1, 10, 15]})
    filtered_cells = filter_cells_by_size(cells=cells, min_size_um=10)
    expected = cells.loc[[1, 2]]
    pd.testing.assert_frame_equal(filtered_cells, expected)


def test_filter_cells_by_size_with_max():
    cells = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "area_um": [1, 10, 15]})
    filtered_cells = filter_cells_by_size(cells=cells, max_size_um=10)
    expected = cells.loc[[0, 1]]
    pd.testing.assert_frame_equal(filtered_cells, expected)


def test_filter_cells_by_size_no_params():
    cells = pd.DataFrame({"x": [1, 2], "y": [1, 2], "area_um": [1, 10]})
    filtered_cells = filter_cells_by_size(cells=cells)
    expected = cells
    pd.testing.assert_frame_equal(filtered_cells, expected)


def test_filter_cells_by_size_with_area_pixels():
    cells = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [1, 2, 3],
            "area_pixels": [1, 10, 15],
            "area_um": [np.nan, np.nan, np.nan],
        }
    )
    filtered_cells = filter_cells_by_size(cells=cells, min_size_um=5, max_size_um=10)
    expected = cells.loc[[1]]
    pd.testing.assert_frame_equal(filtered_cells, expected)
