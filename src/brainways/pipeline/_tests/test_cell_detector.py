from unittest.mock import patch

import numpy as np
import pytest

from brainways.pipeline.brainways_params import CellDetectorParams
from brainways.pipeline.cell_detector import CellDetector, filter_by_cell_size


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


def test_cell_detector_on_small_image():
    cell_detector = CellDetector()

    image = np.random.random((20, 20))
    block_size = 2048

    with patch.object(
        cell_detector.stardist,
        "predict_instances",
        return_value=(np.zeros_like(image), None),
    ) as mock_predict:
        cell_detector.run_cell_detector(
            image=image,
            params=CellDetectorParams(normalizer="none"),
            physical_pixel_sizes=(1, 1),
            block_size=block_size,
        )
        mock_predict.assert_called_once_with(
            image,
            axes="YX",
            normalizer=None,
        )


def test_cell_detector_on_large_image():
    cell_detector = CellDetector()
    image = np.random.random((20, 20))
    block_size = 10

    with patch.object(
        cell_detector.stardist,
        "predict_instances_big",
        return_value=(np.zeros_like(image), None),
    ) as mock_predict_big:
        cell_detector.run_cell_detector(
            image=image,
            params=CellDetectorParams(normalizer="none"),
            physical_pixel_sizes=(1, 1),
            block_size=block_size,
        )
        mock_predict_big.assert_called_once_with(
            image,
            axes="YX",
            block_size=block_size,
            min_overlap=0,
            normalizer=None,
        )
