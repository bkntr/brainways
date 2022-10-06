import abc
from pathlib import Path
from typing import Optional

import pandas as pd

from brainways.project.brainways_project_settings import ProjectDocument


class CellDetectionImporter(abc.ABC):
    def find_cell_detections_file(
        self, root: Path, document: ProjectDocument
    ) -> Optional[Path]:
        raise NotImplementedError()

    def read_cells_file(self, path: Path, document: ProjectDocument) -> pd.DataFrame:
        raise NotImplementedError()
