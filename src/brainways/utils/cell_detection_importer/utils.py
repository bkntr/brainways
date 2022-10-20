from typing import List

from brainways.utils.cell_detection_importer.brainways_cell_detection_importer import (
    BrainwaysCellDetectionsImporter,
)
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from brainways.utils.cell_detection_importer.qupath_cell_detection_importer import (
    QupathCellDetectionsImporter,
)

_CELL_DETECTION_IMPORTERS = {
    "brainways": BrainwaysCellDetectionsImporter,
    "qupath": QupathCellDetectionsImporter,
}


def cell_detection_importer_types() -> List[str]:
    return list(_CELL_DETECTION_IMPORTERS.keys())


def get_cell_detection_importer(name: str) -> CellDetectionImporter:
    return _CELL_DETECTION_IMPORTERS[name]()