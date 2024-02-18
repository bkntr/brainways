import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import click
import napari
import numpy as np
import rpack
import tifffile


def update_layer_contrast_limits(
    layer,
    contrast_limits_quantiles: Tuple[float, float] = (0.01, 0.98),
    contrast_limits_range_quantiles: Tuple[float, float] = (0.0, 1.0),
) -> None:
    nonzero_mask = layer.data > 0
    if (~nonzero_mask).all():
        return

    limit_0, limit_1, limit_range_0, limit_range_1 = np.quantile(
        layer.data[nonzero_mask],
        (*contrast_limits_quantiles, *contrast_limits_range_quantiles),
    )
    layer.contrast_limits = (limit_0, limit_1 + 1e-8)
    layer.contrast_limits_range = (limit_range_0, limit_range_1 + 1e-8)


def create_image_grid(
    images: List[np.ndarray], positions: Optional[List[Tuple[int, int]]] = None
):
    sizes = [(image.shape[1], image.shape[0]) for image in images]
    if positions is None:
        positions = rpack.pack(sizes)

        bbox_size = rpack.bbox_size(sizes, rpack.pack(sizes))
        aspect = max(bbox_size) / min(bbox_size)
        while aspect > 4:
            try:
                max_length = int(max(bbox_size) / 2)
                positions = rpack.pack(
                    sizes, max_width=max_length, max_height=max_length
                )
                bbox_size = rpack.bbox_size(sizes, positions)
                aspect = max(bbox_size) / min(bbox_size)
            except Exception:
                break

    grid_width, grid_height = rpack.bbox_size(sizes, positions)
    grid = np.zeros((grid_height, grid_width), dtype=images[0].dtype)
    for image, position in zip(images, positions):
        x, y = position
        image_h, image_w = image.shape[0], image.shape[1]
        grid[y : y + image_h, x : x + image_w] = image
    return grid, positions


def create_cells_grid(cells_list: List[np.ndarray], positions: List[Tuple[int, int]]):
    grid = [cells + position for cells, position in zip(cells_list, positions)]
    grid = np.concatenate(grid)
    return grid


@click.command()
@click.option(
    "--input",
    type=Path,
    help="Input directory of subjects to display area for.",
)
def display_area(
    input: Path,
):
    images = [
        tifffile.imread(path)
        for path in sorted(input.glob("*.tiff"), key=lambda x: x.name)
    ]
    with open(input / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    grid, positions = create_image_grid(images)
    cells_grid = create_cells_grid(
        [row["cells"] for row in metadata], positions=positions
    )

    viewer = napari.Viewer()
    layer = viewer.add_image(grid, colormap="green")
    update_layer_contrast_limits(layer)
    viewer.add_points(cells_grid[:, ::-1], face_color="red", edge_color="red", size=50)

    image_shapes = [(image.shape[0], image.shape[1]) for image in images]
    rectangles = np.array(
        [((y, x), (y + h, x + w)) for (x, y), (h, w) in zip(positions, image_shapes)]
    )
    viewer.add_shapes(
        rectangles,
        features={"text": [row["text"] for row in metadata]},
        shape_type="rectangle",
        edge_color="transparent",
        face_color="transparent",
        text={
            "string": "{text}",
            "anchor": "upper_left",
            "translation": [0, 0],
            "size": 10,
            "color": "red",
        },
        name="text",
        opacity=1.0,
    )
    if "condition" in metadata:
        viewer.title = metadata["condition"]

    napari.run()


if __name__ == "__main__":
    display_area()
