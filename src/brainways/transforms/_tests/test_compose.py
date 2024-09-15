import numpy as np

from brainways.transforms.base import BrainwaysTransform
from brainways.transforms.compose import Compose


class MockTransform(BrainwaysTransform):
    def __init__(self, value: int = 1):
        self.value = value

    def transform_image(self, image, output_size=None, mode="bilinear"):
        return image + self.value

    def transform_points(self, points):
        return points + self.value

    def inv(self):
        return MockTransform(-self.value)


def test_transform_image():
    transforms = [MockTransform(), MockTransform()]
    compose = Compose(transforms)
    image = np.array([[1, 2], [3, 4]])
    transformed_image = compose.transform_image(image)
    expected_image = np.array([[3, 4], [5, 6]])
    np.testing.assert_array_equal(transformed_image, expected_image)


def test_transform_points():
    transforms = [MockTransform(), MockTransform()]
    compose = Compose(transforms)
    points = np.array([1, 2, 3])
    transformed_points = compose.transform_points(points)
    expected_points = np.array([3, 4, 5])
    np.testing.assert_array_equal(transformed_points, expected_points)


def test_inv():
    transforms = [MockTransform(1), MockTransform(2)]
    compose = Compose(transforms)
    inv_compose = compose.inv()

    # Check that inv_compose is a Compose object
    assert isinstance(inv_compose, Compose)

    # Check that inv_compose has the same number of transforms
    assert len(inv_compose.transforms) == len(transforms)

    # Verify that applying the inverse transforms results in the original input
    image = np.array([[1, 2], [3, 4]])
    transformed_image = compose.transform_image(image)
    reverted_image = inv_compose.transform_image(transformed_image)
    np.testing.assert_array_equal(reverted_image, image)

    points = np.array([1, 2, 3])
    transformed_points = compose.transform_points(points)
    reverted_points = inv_compose.transform_points(transformed_points)
    np.testing.assert_array_equal(reverted_points, points)
