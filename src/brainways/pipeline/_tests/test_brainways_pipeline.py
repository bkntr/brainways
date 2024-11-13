from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from brainways.pipeline.brainways_params import (
    AffineTransform2DParams,
    AtlasRegistrationParams,
    BrainwaysParams,
    TPSTransformParams,
)
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline
from brainways.project.info_classes import RegisteredPixelValues, SliceInfo
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


def test_find_2d_affine_transform(test_data, mock_atlas: BrainwaysAtlas):
    image, _ = test_data
    params = BrainwaysParams(atlas=AtlasRegistrationParams(rot_frontal=10.0))
    pipeline = BrainwaysPipeline(mock_atlas)
    affine_params = pipeline.find_2d_affine_transform(image, params)
    assert affine_params.angle == 10


def test_get_registered_values_on_image_structure_ids():
    # Mock the BrainwaysAtlas
    mock_atlas = MagicMock()
    mock_atlas.shape = (1, 100, 100)

    # Mock the SliceInfo
    mock_slice_info = MagicMock(spec=SliceInfo)
    lowres_image_size = (51, 51)
    image_size = (102, 102)
    mock_slice_info.params = BrainwaysParams(
        atlas=AtlasRegistrationParams(),
        affine=AffineTransform2DParams(angle=90),
    )
    mock_slice_info.lowres_image_size = lowres_image_size
    mock_slice_info.image_size = image_size

    # Mock the atlas slice and its annotation
    mock_atlas_slice = MagicMock()
    mock_atlas_slice.annotation = np.random.randint(0, 255, lowres_image_size)
    expected_annotation = cv2.resize(
        mock_atlas_slice.annotation, image_size, interpolation=cv2.INTER_NEAREST
    )
    expected_annotation = cv2.rotate(
        expected_annotation, cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    # Mock the get_atlas_slice method to return the mocked atlas slice
    pipeline = BrainwaysPipeline(mock_atlas)
    pipeline.get_atlas_slice = MagicMock(return_value=mock_atlas_slice)

    # Call the method under test
    result = pipeline.get_registered_values_on_image(
        mock_slice_info, RegisteredPixelValues.STRUCTURE_IDS
    )

    # Assert the expected behavior/output
    assert result.shape == image_size
    assert result.dtype == np.int64
    assert np.all(result == expected_annotation)


def test_get_registered_values_on_image_pixel_coordinates():
    # Mock the BrainwaysAtlas
    mock_atlas = MagicMock()
    mock_atlas.shape = (1, 100, 100)

    # Mock the SliceInfo
    mock_slice_info = MagicMock(spec=SliceInfo)
    lowres_image_size = (51, 51)
    image_size = (102, 102)
    points = np.random.randint(0, 10, (5, 2))
    mock_slice_info.params = BrainwaysParams(
        atlas=AtlasRegistrationParams(),
        affine=AffineTransform2DParams(angle=90),
        tps=TPSTransformParams(points, points),
    )
    mock_slice_info.lowres_image_size = lowres_image_size
    mock_slice_info.image_size = image_size

    # Mock the get_slice_coordinates function
    mock_coordinates = np.random.randint(0, 255, (51, 51, 3))
    expected_coordinates = cv2.resize(
        mock_coordinates, image_size, interpolation=cv2.INTER_NEAREST
    )
    expected_coordinates = cv2.rotate(
        expected_coordinates, cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    # Mock the get_slice_coordinates method to return the mocked coordinates
    pipeline = BrainwaysPipeline(mock_atlas)

    with patch(
        "brainways.pipeline.brainways_pipeline.get_slice_coordinates",
        return_value=mock_coordinates,
    ):
        # Call the method under test
        result = pipeline.get_registered_values_on_image(
            mock_slice_info, RegisteredPixelValues.PIXEL_COORDINATES
        )

    # Assert the expected behavior/output
    assert result.shape == image_size + (3,)
    assert result.dtype == np.int64
    assert np.all(result == expected_coordinates)


def test_get_registered_values_on_image_micron_coordinates():
    # Mock the BrainwaysAtlas
    mock_atlas = MagicMock()
    mock_atlas.shape = (1, 100, 100)
    mock_atlas.brainglobe_atlas.resolution = (10, 20, 30)

    # Mock the SliceInfo
    mock_slice_info = MagicMock(spec=SliceInfo)
    lowres_image_size = (51, 51)
    image_size = (102, 102)
    points = np.random.randint(0, 10, (5, 2))
    mock_slice_info.params = BrainwaysParams(
        atlas=AtlasRegistrationParams(),
        affine=AffineTransform2DParams(angle=90),
        tps=TPSTransformParams(points, points),
    )
    mock_slice_info.lowres_image_size = lowres_image_size
    mock_slice_info.image_size = image_size

    # Mock the get_slice_coordinates function
    mock_coordinates = np.random.randint(0, 255, (51, 51, 3))
    expected_coordinates = cv2.resize(
        mock_coordinates, image_size, interpolation=cv2.INTER_NEAREST
    )
    expected_coordinates = (
        cv2.rotate(expected_coordinates, cv2.ROTATE_90_COUNTERCLOCKWISE)
        * mock_atlas.brainglobe_atlas.resolution
    )

    # Mock the get_slice_coordinates method to return the mocked coordinates
    pipeline = BrainwaysPipeline(mock_atlas)

    with patch(
        "brainways.pipeline.brainways_pipeline.get_slice_coordinates",
        return_value=mock_coordinates,
    ):
        # Call the method under test
        result = pipeline.get_registered_values_on_image(
            mock_slice_info, RegisteredPixelValues.MICRON_COORDINATES
        )

    # Assert the expected behavior/output
    assert result.shape == image_size + (3,)
    assert result.dtype == np.int64
    assert np.all(result == expected_coordinates)
