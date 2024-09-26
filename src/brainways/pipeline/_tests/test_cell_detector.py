import numpy as np
import pytest

from brainways.pipeline.brainways_params import CellDetectorParams
from brainways.pipeline.cell_detector import filter_by_cell_size


@pytest.fixture
def test_data():
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [3, 3, 0, 0],
            [3, 0, 0, 0],
        ]
    )
    image = np.random.random(labels.shape)
    params = CellDetectorParams(
        cell_size_range=(0, 10), normalizer="none", normalizer_range=(0, 1)
    )
    return labels, image, params


def test_filter_by_cell_size_no_physical_pixel_sizes(test_data):
    labels, image, params = test_data
    physical_pixel_sizes = (np.nan, np.nan)
    filtered_labels = filter_by_cell_size(labels, image, params, physical_pixel_sizes)
    assert filtered_labels.shape == labels.shape
    assert np.array_equal(filtered_labels, labels)


def test_filter_by_cell_size_with_physical_pixel_sizes(test_data):
    labels, image, params = test_data
    physical_pixel_sizes = (0.5, 0.5)
    filtered_labels = filter_by_cell_size(labels, image, params, physical_pixel_sizes)
    assert filtered_labels.shape == labels.shape
    assert np.array_equal(filtered_labels, labels)


def test_filter_by_cell_size_range(test_data):
    labels, image, params = test_data
    params.cell_size_range = (2, 4)
    physical_pixel_sizes = (1, 1)
    filtered_labels = filter_by_cell_size(labels, image, params, physical_pixel_sizes)
    assert filtered_labels.shape == labels.shape
    assert np.array_equal(filtered_labels, labels)


def test_filter_by_cell_size_actual_filtering(test_data):
    labels, image, params = test_data
    params.cell_size_range = (6, 6)
    physical_pixel_sizes = (2, 1)
    filtered_labels = filter_by_cell_size(labels, image, params, physical_pixel_sizes)
    expected_labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [3, 3, 0, 0],
            [3, 0, 0, 0],
        ]
    )
    assert filtered_labels.shape == labels.shape
    assert np.array_equal(filtered_labels, expected_labels)


def test_filter_by_cell_size_actual_filtering_pixels(test_data):
    labels, image, params = test_data
    params.cell_size_range = (3, 3)
    physical_pixel_sizes = (np.nan, np.nan)
    filtered_labels = filter_by_cell_size(labels, image, params, physical_pixel_sizes)
    expected_labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [3, 3, 0, 0],
            [3, 0, 0, 0],
        ]
    )
    assert filtered_labels.shape == labels.shape
    assert np.array_equal(filtered_labels, expected_labels)
