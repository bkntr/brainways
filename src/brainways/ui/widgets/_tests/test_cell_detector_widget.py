from unittest.mock import Mock, patch

import pytest

from brainways.project.info_classes import MaskFileFormat, SliceSelection
from brainways.ui.controllers.cell_detector_controller import CellDetectorController
from brainways.ui.widgets.cell_detector_widget import CellDetectorWidget


@pytest.fixture
def cell_detector_widget():
    controller = CellDetectorController(Mock())
    with patch.object(controller, "run_cell_detector_async"):
        yield CellDetectorWidget(controller)


@pytest.mark.parametrize(
    "slice_selection",
    [
        pytest.param(SliceSelection.CURRENT_SLICE.value, id="current_slice"),
        pytest.param(SliceSelection.ALL_SUBJECTS.value, id="all_subjects"),
    ],
)
@pytest.mark.parametrize(
    "resume",
    [pytest.param(True, id="resume_true"), pytest.param(False, id="resume_false")],
)
@pytest.mark.parametrize(
    "save_cell_detection_masks_file_format",
    [
        pytest.param(MaskFileFormat.MAT.value, id="mat"),
        pytest.param(MaskFileFormat.NPZ.value, id="npz"),
        pytest.param("Don't Save", id="dont_save"),
    ],
)
def test_run_cell_detector(
    cell_detector_widget,
    slice_selection: str,
    resume: bool,
    save_cell_detection_masks_file_format: str,
):
    with patch(
        "brainways.ui.widgets.cell_detector_widget.request_values"
    ) as mock_request_values, patch(
        "brainways.ui.widgets.cell_detector_widget.show_warning_dialog"
    ) as mock_show_warning_dialog:
        mock_request_values.return_value = {
            "slice_selection": slice_selection,
            "resume": resume,
            "save_cell_detection_masks_file_format": save_cell_detection_masks_file_format,
        }
        mock_show_warning_dialog.return_value = True
        cell_detector_widget.run_cell_detector()
        expected_file_format = (
            None
            if save_cell_detection_masks_file_format == "Don't Save"
            else MaskFileFormat(save_cell_detection_masks_file_format)
        )
        cell_detector_widget.controller.run_cell_detector_async.assert_called_once_with(
            slice_selection=SliceSelection(slice_selection),
            resume=resume,
            save_cell_detection_masks_file_format=expected_file_format,
        )


def test_run_cell_detector_resume_false_cancelled(cell_detector_widget):
    with patch(
        "brainways.ui.widgets.cell_detector_widget.request_values"
    ) as mock_request_values, patch(
        "brainways.ui.widgets.cell_detector_widget.show_warning_dialog"
    ) as mock_show_warning_dialog:
        mock_request_values.return_value = {
            "slice_selection": SliceSelection.CURRENT_SLICE.value,
            "resume": False,
            "save_cell_detection_masks_file_format": "Don't Save",
        }
        mock_show_warning_dialog.return_value = False
        cell_detector_widget.run_cell_detector()
        cell_detector_widget.controller.run_cell_detector_async.assert_not_called()


def test_run_cell_detector_request_values_none(cell_detector_widget):
    with patch(
        "brainways.ui.widgets.cell_detector_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = None
        cell_detector_widget.run_cell_detector()
        cell_detector_widget.controller.run_cell_detector_async.assert_not_called()
