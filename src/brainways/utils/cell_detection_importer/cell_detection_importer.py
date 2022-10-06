import abc
from pathlib import Path
from typing import Optional

import numpy as np

from brainways.project.brainways_project_settings import ProjectDocument


class CellDetectionImporter(abc.ABC):
    def find_cell_detections_file(
        self, root: Path, document: ProjectDocument
    ) -> Optional[Path]:
        raise NotImplementedError()

    def read_cells_file(self, path: Path, document: ProjectDocument) -> np.ndarray:
        raise NotImplementedError()
