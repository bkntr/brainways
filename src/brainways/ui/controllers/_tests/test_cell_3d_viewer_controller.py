import numpy as np

from brainways.pipeline.tps import TPSTransformParams
from brainways.project.info_classes import BrainwaysParams
from brainways.ui.controllers.cell_3d_viewer_controller import Cell3DViewerController


def test_enabled_false_by_default():
    params = BrainwaysParams()
    enabled = Cell3DViewerController.enabled(params)
    assert enabled is False


def test_enabled_with_tps_params():
    params = BrainwaysParams(
        tps=TPSTransformParams(
            points_src=np.array([10, 10]), points_dst=np.array([10, 10])
        )
    )
    enabled = Cell3DViewerController.enabled(params)
    assert enabled is True
