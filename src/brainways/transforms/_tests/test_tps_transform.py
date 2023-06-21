import numpy as np
import numpy.testing as npt

from brainways.pipeline.brainways_params import TPSTransformParams
from brainways.transforms.tps_transform import TPSTransform


def test_tps_transform_points():
    points_src = np.random.randint(0, 10, (5, 2))
    points_dst = points_src + np.random.randint(0, 2, (5, 2))
    params = TPSTransformParams(points_src.tolist(), points_dst.tolist())
    transform = TPSTransform(params)
    points_transformed = transform.transform_points(points_src)
    npt.assert_allclose(points_transformed, points_dst, rtol=1e-4, atol=1e-3)


def test_tps_transform_image_doesnt_change_image_on_same_points():
    points = np.random.rand(5, 2).astype(np.float32) * 5
    image = np.random.rand(5, 5).astype(np.float32)
    params = TPSTransformParams(points, points)
    transform = TPSTransform(params)
    image_transformed = transform.transform_image(image)
    npt.assert_allclose(image, image_transformed, rtol=1e-6)


def test_tps_transform_image_warps_correctly():
    image = np.random.rand(30, 40).astype(np.float32)
    points_src = np.array([[0, 0], [0, 29], [39, 0], [39, 29], [20, 15]]).astype(
        np.float32
    )
    points_dst = np.array([[0, 0], [0, 29], [39, 0], [39, 29], [24, 15]]).astype(
        np.float32
    )

    params = TPSTransformParams(points_src, points_dst)
    transform = TPSTransform(params)
    image_transformed = transform.transform_image(image)

    value_src = image[int(points_src[-1, 1]), int(points_src[-1, 0])]
    value_dst = image_transformed[int(points_dst[-1, 1]), int(points_dst[-1, 0])]

    npt.assert_allclose(value_src, value_dst, rtol=1e-4)


def test_tps_transform_points_inv():
    points_src = np.random.randint(0, 10, (5, 2))
    points_dst = points_src + np.random.randint(0, 2, (5, 2))
    params = TPSTransformParams(points_src, points_dst)
    transform = TPSTransform(params)
    transform_inv = transform.inv()
    points_transformed = transform_inv.transform_points(
        transform.transform_points(points_src)
    )
    npt.assert_allclose(points_transformed, points_src, rtol=1e-4, atol=1e-3)


def test_tps_transform_empty_points():
    points = np.random.rand(5, 2).astype(np.float32) * 5
    params = TPSTransformParams(points, points)
    transform = TPSTransform(params)
    transformed_points = transform.transform_points(np.empty((0, 2)))
    assert transformed_points.shape == (0, 2)


def test_tps_transform_points_dtype():
    points = np.random.randint(0, 10, (5, 2))
    params = TPSTransformParams(points, points)
    transform = TPSTransform(params)
    points_transformed = transform.transform_points(points)
    assert points_transformed.dtype == np.float32
