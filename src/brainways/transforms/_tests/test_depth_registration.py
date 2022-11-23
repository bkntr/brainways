import numpy as np
import numpy.testing as npt
import torch

from brainways.transforms.depth_registration import (
    DepthRegistration,
    DepthRegistrationParams,
)


def test_depth_registration_transform_points():
    params = DepthRegistrationParams(td=-1, rx=0, ry=0)
    transform = DepthRegistration(params, volume_shape=(10, 10, 10))
    points = np.array([[0, 0]])
    points_transformed = transform.transform_points(points)
    expected = np.array([[0, 0, 1]])
    npt.assert_allclose(points_transformed, expected)


def test_depth_registration_slice_volume():
    params = DepthRegistrationParams(td=-1, rx=0, ry=0)
    transform = DepthRegistration(params, volume_shape=(3, 3, 3))
    volume = torch.rand(3, 3, 3)
    slice = transform.slice_volume(volume)
    expected = volume[1]
    npt.assert_allclose(slice, expected)


def test_depth_registration_slice_volume_rx():
    params = DepthRegistrationParams(td=-1, rx=90, ry=0)
    transform = DepthRegistration(params, volume_shape=(3, 3, 3))
    volume = torch.rand(3, 3, 3)
    slice = transform.slice_volume(volume)
    expected = volume[:, 1, :]
    npt.assert_allclose(slice, expected, rtol=1e-4)


def test_depth_registration_slice_volume_ry():
    params = DepthRegistrationParams(td=-1, rx=0, ry=90)
    transform = DepthRegistration(params, volume_shape=(3, 3, 3))
    volume = torch.rand(3, 3, 3)
    slice = transform.slice_volume(volume)
    npt.assert_allclose(slice[0, 0], volume[2, 0, 1], rtol=1e-5)
    npt.assert_allclose(slice[2, 2], volume[0, 2, 1], rtol=1e-5)


def test_depth_registration_empty_points():
    params = DepthRegistrationParams()
    transform = DepthRegistration(params, volume_shape=(3, 3, 3))
    transformed_points = transform.transform_points(np.empty((0, 2)))
    assert transformed_points.shape == (0, 3)
