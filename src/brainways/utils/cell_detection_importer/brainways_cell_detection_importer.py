from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from brainways.project.brainways_project_settings import ProjectDocument
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)


class BrainwaysCellDetectionsImporter(CellDetectionImporter):
    def find_cell_detections_file(
        self, root: Path, document: ProjectDocument
    ) -> Optional[Path]:
        csv_filename = (
            f"{Path(document.path.filename).stem}_scene{document.path.scene}.csv"
        )
        csv_path = root / csv_filename
        if csv_path.exists():
            return csv_path
        else:
            return None

    def read_cells_file(self, path: Path, document: ProjectDocument) -> np.ndarray:
        cells_df = pd.read_csv(path)
        cells = cells_df[["centroid-1", "centroid-0"]].to_numpy()
        if (cells > 1).any():
            cells = cells / document.image_size[::-1]
        return cells
