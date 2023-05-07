from pathlib import Path

import click
from tqdm import tqdm

from brainways.project.brainways_project import BrainwaysProject

# if NAPARI_AVAILABLE:
#     import napari
#
# # CONTRAST_LIMITS = (12000, 50000)
# CFOS_CONTRAST_LIMITS = (99.5, 99.97)
# DAPI_CONTRAST_LIMITS = (0, 98)
#
#
# def display_results(
#     cfos: np.ndarray, labels: np.ndarray, dapi: Optional[np.ndarray] = None
# ):
#     if not NAPARI_AVAILABLE:
#         raise ImportError(
#             "Please install napari to display results: "
#             "`pip install napari` or `pip install brainways[all]`"
#         ) from None
#     viewer = napari.Viewer()
#     cfos_layer = viewer.add_image(cfos, colormap="green")
#     cfos_layer.reset_contrast_limits_range()
#     cfos_layer.contrast_limits = np.percentile(cfos.flat, CFOS_CONTRAST_LIMITS)
#     if dapi is not None:
#         dapi_layer = viewer.add_image(dapi, colormap="blue", blending="additive")
#         dapi_layer.reset_contrast_limits_range()
#         dapi_layer.contrast_limits = np.percentile(dapi.flat, DAPI_CONTRAST_LIMITS)
#     viewer.add_labels(labels)
#     napari.run()
#
#
# def run_cell_detector(
#     image_path: ImagePath,
#     cell_detector: CellDetector,
#     channel: int,
#     output_dir: Path,
#     dapi_channel: Optional[int] = None,
#     display: bool = False,
# ):
#     output_filename = output_dir / (
#         Path(image_path.filename).stem + f"_scene{image_path.scene}.csv"
#     )
#     print(output_filename)
#     if output_filename.exists() and not display:
#         return
#
#     reader = QupathReader(image_path.filename)
#     reader.set_scene(image_path.scene)
#     image = reader.get_image_data("YX", C=channel)
#     # min_val, max_val = np.percentile(image, CFOS_CONTRAST_LIMITS)
#     # print(min_val, max_val)
#     # normalizer = MinMaxNormalizer(min=min_val, max=max_val)
#     normalizer = ClaheNormalizer()
#     labels = cell_detector.run_cell_detector(image, normalizer=normalizer)
#     cells = cell_detector.cells(labels, image)
#
#     if display:
#         dapi = None
#         if dapi_channel is not None:
#             dapi = reader.get_image_data("YX", C=dapi_channel)
#         display_results(cfos=image, dapi=dapi, labels=labels)
#     else:
#         cells.to_csv(output_filename)


@click.command()
@click.option(
    "--brainways-path",
    type=Path,
    required=True,
    help="Brainways project path.",
)
@click.option(
    "--channel",
    type=int,
    default=0,
    show_default=True,
    help="Channel index to detect in the images.",
)
@click.option(
    "--dapi-channel",
    type=int,
    help="DAPI channel index in the image (optional).",
)
@click.option(
    "--display", is_flag=True, help="Display detected cells (requires napari)."
)
def cell_detection(
    brainways_path: Path, channel: int, dapi_channel: int, display: bool
):
    project = BrainwaysProject.open(brainways_path)
    for _ in tqdm(project.run_cell_detector_iter(), total=project.n_valid_images):
        _


if __name__ == "__main__":
    cell_detection()
