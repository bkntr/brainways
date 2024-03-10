from pathlib import Path
from typing import Optional

import pandas as pd

from brainways.project.info_classes import SliceInfo
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)


class KerenCellDetectionsImporter(CellDetectionImporter):
    parameters = {
        "cfos_threshold": int,
        "drd1_threshold": int,
        "drd2_threshold": int,
        "oxtr_threshold": int,
    }

    def __init__(
        self,
        cfos_threshold: int,
        drd1_threshold: int,
        drd2_threshold: int,
        oxtr_threshold: int,
    ):
        super().__init__()
        self.cfos_threshold = cfos_threshold
        self.drd1_threshold = drd1_threshold
        self.drd2_threshold = drd2_threshold
        self.oxtr_threshold = oxtr_threshold

    def find_cell_detections_file(
        self,
        root: Path,
        document: SliceInfo,
    ) -> Optional[Path]:
        csv_filename = f"{Path(document.path.filename).name} Detections.txt"
        csv_path = root / csv_filename
        if csv_path.exists():
            return csv_path
        else:
            return None

    def read_cells_file(self, path: Path, document: SliceInfo) -> pd.DataFrame:
        input_cells_df = pd.read_csv(path, sep="\t")
        input_cells_df = input_cells_df[
            input_cells_df["Class"].isin(("Positive", "Negative"))
        ]
        image_size_um = [
            document.image_size[1] * document.physical_pixel_sizes[1],
            document.image_size[0] * document.physical_pixel_sizes[0],
        ]
        brainways_cells_df = pd.DataFrame(
            {
                "x": input_cells_df["Centroid X µm"] / image_size_um[0],
                "y": input_cells_df["Centroid Y µm"] / image_size_um[1],
                "LABEL-cFos": input_cells_df["Subcellular: Channel 5: Num single spots"]
                > self.cfos_threshold,
                "LABEL-Drd1": input_cells_df["Subcellular: Channel 2: Num single spots"]
                > self.drd1_threshold,
                "LABEL-Drd2": input_cells_df["Subcellular: Channel 3: Num single spots"]
                > self.drd2_threshold,
                "LABEL-Oxtr": input_cells_df["Subcellular: Channel 4: Num single spots"]
                > self.oxtr_threshold,
            }
        ).dropna()
        assert (brainways_cells_df[["x", "y"]].values < 1).all()
        return brainways_cells_df
