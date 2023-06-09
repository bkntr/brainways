import numpy as np

from brainways.pipeline.cell_detector import CellDetector


def test_cell_detector_cells_missing_physical_pixel_sizes():
    labels = image = np.zeros((50, 50), dtype=np.uint8)
    cells_df = CellDetector().cells(
        labels=labels, image=image, physical_pixel_sizes=(float("nan"), float("nan"))
    )
    assert cells_df["area_um"].isna().all()
