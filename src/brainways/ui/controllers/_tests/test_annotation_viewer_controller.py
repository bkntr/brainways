import numpy as np

from brainways.pipeline.tps import TPSTransformParams
from brainways.project.info_classes import BrainwaysParams
from brainways.ui.controllers.annotation_viewer_controller import (
    AnnotationViewerController,
)


def test_enabled_false_by_default():
    params = BrainwaysParams()
    enabled = AnnotationViewerController.enabled(params)
    assert enabled is False


def test_enabled_with_tps_params():
    params = BrainwaysParams(
        tps=TPSTransformParams(
            points_src=np.array([10, 10]), points_dst=np.array([10, 10])
        )
    )
    enabled = AnnotationViewerController.enabled(params)
    assert enabled is True
