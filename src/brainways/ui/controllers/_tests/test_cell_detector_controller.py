from unittest.mock import MagicMock

from brainways.project.info_classes import BrainwaysParams, SliceSelection
from brainways.ui.controllers.cell_detector_controller import CellDetectorController


def test_enabled():
    params = BrainwaysParams()
    enabled = CellDetectorController.enabled(params)
    assert enabled is True


def test_run_cell_detector_async():
    ui_mock = MagicMock()
    controller = CellDetectorController(ui_mock)
    slice_selection = SliceSelection.ALL_SUBJECTS
    resume = True

    slice_infos = ["slice1", "slice2"]
    ui_mock.get_slice_selection.return_value = slice_infos

    controller.run_cell_detector_async(slice_selection, resume)
    ui_mock.do_work_async.assert_called_once_with(
        ui_mock.project.run_cell_detector_iter,
        slice_infos=slice_infos,
        resume=resume,
        progress_label="Running Cell Detector on All Subjects...",
        progress_max_value=len(slice_infos),
    )
