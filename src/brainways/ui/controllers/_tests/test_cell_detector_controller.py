from unittest.mock import MagicMock

from brainways.project.info_classes import (
    BrainwaysParams,
    MaskFileFormat,
    SliceSelection,
)
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
    file_format = MaskFileFormat.CSV

    slice_infos = ["slice1", "slice2"]
    ui_mock.get_slice_selection.return_value = slice_infos

    controller.run_cell_detector_async(slice_selection, resume, file_format)
    ui_mock.do_work_async.assert_called_once_with(
        ui_mock.project.run_cell_detector_iter,
        slice_infos=slice_infos,
        resume=resume,
        save_cell_detection_masks_file_format=file_format,
        progress_label="Running Cell Detector on All Subjects...",
        progress_max_value=len(slice_infos),
    )
