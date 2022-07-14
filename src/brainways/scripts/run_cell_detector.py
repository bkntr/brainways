import argparse
from pathlib import Path
from typing import Optional

import napari
import numpy as np
from duracell.ui.model.cell_detector_model import CellDetectorModel, MinMaxNormalizer
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject
from brainways.utils.io import ImagePath
from brainways.utils.io.readers import get_reader, get_scenes

# CONTRAST_LIMITS = (12000, 50000)
CFOS_CONTRAST_LIMITS = (99.5, 99.97)
DAPI_CONTRAST_LIMITS = (0, 98)


def display_results(
    cfos: np.ndarray, labels: np.ndarray, dapi: Optional[np.ndarray] = None
):
    viewer = napari.Viewer()
    cfos_layer = viewer.add_image(cfos, colormap="green")
    cfos_layer.reset_contrast_limits_range()
    cfos_layer.contrast_limits = np.percentile(cfos.flat, CFOS_CONTRAST_LIMITS)
    if dapi is not None:
        dapi_layer = viewer.add_image(dapi, colormap="blue", blending="additive")
        dapi_layer.reset_contrast_limits_range()
        dapi_layer.contrast_limits = np.percentile(dapi.flat, DAPI_CONTRAST_LIMITS)
    viewer.add_labels(labels)
    napari.run()


def run_cell_detector(
    image_path: ImagePath,
    cell_detector: CellDetectorModel,
    cfos_channel: int,
    output_dir: Path,
    dapi_channel: Optional[int] = None,
    display: bool = False,
):
    output_filename = output_dir / (
        Path(image_path.filename).stem + f"_scene{image_path.scene}.csv"
    )
    print(output_filename)
    if output_filename.exists() and not display:
        return

    reader = get_reader(path=image_path)
    image = reader.read_image(channel=cfos_channel, scale=1)
    min_val, max_val = np.percentile(image, CFOS_CONTRAST_LIMITS)
    print(min_val, max_val)
    normalizer = MinMaxNormalizer(min=min_val, max=max_val)
    labels = cell_detector.run_cell_detector(image, normalizer=normalizer)
    cells = cell_detector.cells(labels, image)

    if display:
        dapi = None
        if dapi_channel is not None:
            dapi = reader.read_image(channel=dapi_channel, scale=1)
        display_results(cfos=image, dapi=dapi, labels=labels)
    else:
        cells.to_csv(output_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cfos-channel", type=int, default=0)
    parser.add_argument("--dapi-channel", type=int)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)

    output.mkdir(parents=True, exist_ok=True)

    cell_detector = CellDetectorModel()

    project_paths = sorted(list(args.input.glob("*")))
    with tqdm(project_paths) as t:
        for path in t:
            try:
                if (path / "brainways.bin").exists():
                    project = BrainwaysProject.open(path / "brainways.bin")
                    image_paths = [
                        document.path for i, document in project.valid_documents
                    ]
                else:
                    image_paths = [
                        ImagePath(path, scene=scene_i)
                        for scene_i in range(len(get_scenes(path)))
                    ]
            except Exception as e:
                print(e)
                continue

            for image_path in image_paths:
                try:
                    run_cell_detector(
                        image_path=image_path,
                        cell_detector=cell_detector,
                        cfos_channel=args.cfos_channel,
                        output_dir=args.output,
                        dapi_channel=args.dapi_channel,
                        display=args.display,
                    )
                except Exception as e:
                    t.write(f"Error in file {image_path}: {e}")


if __name__ == "__main__":
    main()
