import numpy as np
import numpy.testing

from brainways.pipeline.brainways_params import AffineTransform2DParams
from brainways.transforms.affine_transform_2d import BrainwaysAffineTransform2D


def test_affine_transform_keypoints():
    params = AffineTransform2DParams(angle=0, tx=1, ty=0, sx=1, sy=1)
    transform = BrainwaysAffineTransform2D(params=params, input_size=(10, 10))
    points = np.array([[0, 0]])
    points_transformed = transform.transform_points(points)
    expected = np.array([[1, 0]])
    np.testing.assert_allclose(points_transformed, expected)


def test_affine_transform_image():
    params = AffineTransform2DParams(angle=0, tx=1, ty=0, sx=1, sy=1)
    transform = BrainwaysAffineTransform2D(params=params, input_size=(3, 3))
    image = np.random.rand(3, 3).astype(np.float32)
    image_transformed = transform.transform_image(image)
    expected = np.zeros_like(image)
    expected[:, 1:] = image[:, :-1]
    np.testing.assert_allclose(image_transformed, expected)


def test_affine_transform_multichannel_image():
    params = AffineTransform2DParams(angle=0, tx=1, ty=0, sx=1, sy=1)
    transform = BrainwaysAffineTransform2D(params=params, input_size=(3, 3))
    image = np.random.rand(3, 3, 5).astype(np.float32)
    image_transformed = transform.transform_image(image)
    expected = np.zeros_like(image)
    expected[:, 1:] = image[:, :-1]
    np.testing.assert_allclose(image_transformed, expected)


def test_affine_transform_scale_center():
    params = AffineTransform2DParams(angle=0, tx=0, ty=0, sx=0.5, sy=1.0, cx=1, cy=2)
    transform = BrainwaysAffineTransform2D(params=params, input_size=(5, 5))
    points = np.array([[1, 2], [0, 0]])
    points_transformed = transform.transform_points(points)
    expected = np.array([[1, 2], [0.5, 0]], np.float32)
    np.testing.assert_allclose(points_transformed, expected)


def test_affine_transform_inv_points():
    params = AffineTransform2DParams(
        angle=10, tx=10, ty=20, sx=1.5, sy=2.0, cx=5, cy=10
    )
    transform = BrainwaysAffineTransform2D(params=params, input_size=(5, 5))
    transform_inv = transform.inv()
    points = np.array([[1, 2], [3, 4]])
    points_transformed = transform_inv.transform_points(
        transform.transform_points(points)
    )
    numpy.testing.assert_allclose(points, points_transformed, rtol=1e-6)


def test_affine_transform_empty_points():
    params = AffineTransform2DParams()
    transform = BrainwaysAffineTransform2D(params=params, input_size=(5, 5))
    transformed_points = transform.transform_points(np.empty((0, 2)))
    assert transformed_points.shape == (0, 2)
