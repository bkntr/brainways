from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from jsonargparse import auto_cli

from brainways.model.dataset_utils import read_image
from brainways.model.model_utils import load_model
from brainways.model.siamese.siamese_model import SiameseModel


def predict_single(
    image_path: Path, model: SiameseModel, atlas_name: str = "allen_mouse_25um"
) -> None:
    """
    Predict using the model on a single image.

    Args:
        image_path: Path to the image file
        model: Loaded SiameseModel
        atlas_name: Name of the atlas to use for prediction
    """
    # Read image
    image = read_image(str(image_path))

    # Make prediction
    pred_ap, pred_slice = model.predict(image, atlas_name=atlas_name)

    # Visualize results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(pred_slice[0].cpu().numpy(), cmap="gray")
    plt.title("Predicted Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    plt.tight_layout()
    plt.show()


def process_directory(
    input_dir: Path,
    model: SiameseModel,
    start_index: int = 0,
    max_images: int | None = None,
) -> None:
    """
    Process all images in a directory.

    Args:
        input_dir: Directory containing images
        model: Loaded SiameseModel
        start_index: Index to start processing from
        max_images: Maximum number of images to process (None for all)
    """
    image_paths = list(input_dir.rglob("*.jpg"))

    for i, image_path in enumerate(image_paths):
        if i < start_index:
            continue

        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        predict_single(image_path, model)

        if max_images is not None and i >= start_index + max_images - 1:
            break


def main(
    model_dir: str, input: str, start_index: int = 0, max_images: int | None = None
) -> None:
    """
    Main function to run predictions.

    Args:
        model_dir: Directory with model files
        input: Input image or directory
        start_index: Index to start processing from
        max_images: Maximum number of images to process
    """
    # Load the model
    model = load_model(Path(model_dir))

    # Process input
    input_path = Path(input)
    if input_path.is_file():
        predict_single(input_path, model)
    elif input_path.is_dir():
        process_directory(input_path, model, start_index, max_images)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
