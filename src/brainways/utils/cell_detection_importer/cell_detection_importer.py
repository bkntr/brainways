import abc
from pathlib import Path
from typing import Optional

import pandas as pd

from brainways.project.info_classes import SliceInfo


class CellDetectionImporter(abc.ABC):
    parameters = {}

    def __init__(self, **kwargs):
        pass

    def find_cell_detections_file(
        self, root: Path, document: SliceInfo
    ) -> Optional[Path]:
        raise NotImplementedError()

    def read_cells_file(self, path: Path, document: SliceInfo) -> pd.DataFrame:
        raise NotImplementedError()
