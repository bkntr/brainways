from brainways.project.info_classes import BrainwaysParams
from napari_brainways.controllers.cell_detector_controller import CellDetectorController


def test_enabled():
    params = BrainwaysParams()
    enabled = CellDetectorController.enabled(params)
    assert enabled is True
