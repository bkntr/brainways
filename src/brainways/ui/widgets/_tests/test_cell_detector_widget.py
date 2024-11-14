from unittest.mock import Mock, patch

import pytest

from brainways.project.info_classes import SliceSelection
from brainways.ui.controllers.cell_detector_controller import CellDetectorController
from brainways.ui.widgets.cell_detector_widget import CellDetectorWidget


@pytest.fixture
def cell_detector_widget():
    controller = CellDetectorController(Mock())
    with patch.object(controller, "run_cell_detector_async"):
        yield CellDetectorWidget(controller)


def test_run_cell_detector_current_slice(cell_detector_widget):
    with patch(
        "brainways.ui.widgets.cell_detector_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "slice_selection": SliceSelection.CURRENT_SLICE.value,
            "resume": True,
        }
        cell_detector_widget.run_cell_detector()
        cell_detector_widget.controller.run_cell_detector_async.assert_called_once_with(
            slice_selection=SliceSelection.CURRENT_SLICE,
            resume=True,
        )


def test_run_cell_detector_all_subjects(cell_detector_widget):
    with patch(
        "brainways.ui.widgets.cell_detector_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "slice_selection": SliceSelection.ALL_SUBJECTS.value,
            "resume": True,
        }
        cell_detector_widget.run_cell_detector()
        cell_detector_widget.controller.run_cell_detector_async.assert_called_once_with(
            slice_selection=SliceSelection.ALL_SUBJECTS,
            resume=True,
        )


def test_run_cell_detector_resume_false_confirmed(cell_detector_widget):
    with patch(
        "brainways.ui.widgets.cell_detector_widget.request_values"
    ) as mock_request_values, patch(
        "brainways.ui.widgets.cell_detector_widget.show_warning_dialog"
    ) as mock_show_warning_dialog:
        mock_request_values.return_value = {
            "slice_selection": SliceSelection.CURRENT_SLICE.value,
            "resume": False,
        }
        mock_show_warning_dialog.return_value = True
        cell_detector_widget.run_cell_detector()
        cell_detector_widget.controller.run_cell_detector_async.assert_called_once_with(
            slice_selection=SliceSelection.CURRENT_SLICE,
            resume=False,
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
