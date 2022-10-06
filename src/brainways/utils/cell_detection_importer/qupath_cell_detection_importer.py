from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from brainways.project.brainways_project_settings import ProjectDocument
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from brainways.utils.io_utils.readers import QupathReader


class QupathCellDetectionsImporter(CellDetectionImporter):
    def find_cell_detections_file(
        self, root: Path, document: ProjectDocument
    ) -> Optional[Path]:
        csv_filename = f"{Path(document.path.filename).name} Detections.txt"
        csv_path = root / csv_filename
        if csv_path.exists():
            return csv_path
        else:
            return None

    def read_cells_file(self, path: Path, document: ProjectDocument) -> np.ndarray:
        reader = QupathReader(document.path.filename)
        reader.set_scene(document.path.scene)
        cells_df = pd.read_csv(path, sep="\t")
        cfos_cells = cells_df["Subcellular: Channel 5: Num single spots"] > 10
        cells = cells_df.loc[cfos_cells, ["Centroid X µm", "Centroid Y µm"]].values
        image_size_um = [
            reader.dims.X * reader.physical_pixel_sizes.X,
            reader.dims.Y * reader.physical_pixel_sizes.Y,
        ]
        cells = cells / image_size_um
        assert (cells < 1).all()
        return cells
