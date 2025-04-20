import cv2
import numpy as np
import torch
from albumentations.core.composition import TransformType
from brainglobe_atlasapi import BrainGlobeAtlas
from numpy.typing import NDArray


def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from the given file path and return it as a NumPy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = (
            np.dot(image, [0.2989, 0.5870, 0.1140])
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
    return image


def load_atlas_reference(atlas_name: str) -> NDArray[np.float32]:
    """
    Load the atlas reference volume for a given atlas name.

    Parameters:
        atlas_name (str): The name of the atlas.

    Returns:
        NDArray[np.float32]: The loaded atlas reference as a NumPy array of type np.float32 with shape (z, y, x).
    """
    bg_atlas = BrainGlobeAtlas(atlas_name)
    atlas_reference = (bg_atlas.reference / bg_atlas.reference.max()).astype(np.float32)
    return atlas_reference


def transform_atlas_volume(
    atlas_volume: NDArray[np.float32], transform: TransformType
) -> torch.Tensor:
    """
    Transform an atlas volume using the provided transform.

    Args:
        atlas_volume: The atlas volume to transform
        transform: The transform to apply

    Returns:
        Transformed atlas volume as a torch.Tensor
    """
    transformed_atlas_volume = torch.stack(
        [transform(image=atlas_slice)["image"] for atlas_slice in atlas_volume]
    )
    return transformed_atlas_volume
