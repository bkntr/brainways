import numpy as np
import numpy.testing
import pytest

from brainways.pipeline.brainways_params import (
    AffineTransform2DParams,
    TPSTransformParams,
)
from brainways.transforms.affine_transform_2d import BrainwaysAffineTransform2D
from brainways.transforms.depth_registration import (
    DepthRegistration,
    DepthRegistrationParams,
)
from brainways.transforms.image_to_atlas_transform import ImageToAtlasTransform
from brainways.transforms.tps_transform import TPSTransform


@pytest.fixture
def image_to_atlas_transform() -> ImageToAtlasTransform:
    tps_points_src = np.random.randint(0, 10, (10, 2))
    tps_points_dst = tps_points_src.copy()
    tps_points_dst[0, 0] += 1
    return ImageToAtlasTransform(
        atlas_transform=DepthRegistration(
            DepthRegistrationParams(), volume_shape=(10, 10, 10)
        ),
        affine_2d_transform=BrainwaysAffineTransform2D(
            AffineTransform2DParams(tx=10, ty=10), input_size=(64, 64)
        ),
        tps_transform=TPSTransform(
            TPSTransformParams(points_src=tps_points_src, points_dst=tps_points_dst)
        ),
    )


@pytest.mark.skip
def test_image_to_atlas_transform_inv_points(
    image_to_atlas_transform: ImageToAtlasTransform,
):
    transform_inv = image_to_atlas_transform.inv()
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    image_t = image_to_atlas_transform.transform_image(image)
    transform_inv.transform_image(image_t)
    # import matplotlib.pyplot as plt
    #
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.imshow(image / 255)
    # ax2.imshow(image_t), ax3.imshow(image_tt)
    # plt.show()
    # numpy.testing.assert_allclose(image / 255, image_tt, rtol=1e-6)


def test_image_to_atlas_transform_empty_points(
    image_to_atlas_transform: ImageToAtlasTransform,
):
    empty_points = np.empty((0, 2))
    empty_points_t = image_to_atlas_transform.transform_points(empty_points)
    assert empty_points_t.shape == (0, 3)
    assert empty_points_t.dtype == empty_points.dtype
