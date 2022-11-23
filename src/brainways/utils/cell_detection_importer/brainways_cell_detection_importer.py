from pathlib import Path
from typing import Optional

import pandas as pd

from brainways.project.info_classes import SliceInfo
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)


class BrainwaysCellDetectionsImporter(CellDetectionImporter):
    def find_cell_detections_file(
        self, root: Path, document: SliceInfo
    ) -> Optional[Path]:
        csv_filename = (
            f"{Path(document.path.filename).stem}_scene{document.path.scene}.csv"
        )
        csv_path = root / csv_filename
        if csv_path.exists():
            return csv_path
        else:
            return None

    def read_cells_file(self, path: Path, document: SliceInfo) -> pd.DataFrame:
        input_cells_df = pd.read_csv(path)
        label_columns = {
            c: input_cells_df[c]
            for c in input_cells_df.columns
            if c.startswith("LABEL-")
        }
        output_cells_df = pd.DataFrame(
            {
                "x": input_cells_df["centroid-1"],
                "y": input_cells_df["centroid-0"],
                **label_columns,
            }
        )
        if (output_cells_df > 1).any(axis=None):
            output_cells_df["x"] /= document.image_size[1]
            output_cells_df["y"] /= document.image_size[0]
        return output_cells_df
