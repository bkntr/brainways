from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from brainways.project.info_classes import SliceSelection
from napari_brainways.widgets.analysis_widget import AnalysisWidget


@pytest.fixture
def mock_controller():
    controller = Mock()
    return controller


@pytest.fixture
def analysis_widget(mock_controller):
    return AnalysisWidget(mock_controller)


def test_on_export_registration_masks_clicked_current_slice(analysis_widget):
    with patch(
        "napari_brainways.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "output_path": Path("/fake/path"),
            "slice_selection": "Current Slice",
        }
        analysis_widget.on_export_registration_masks_clicked()
        analysis_widget.controller.export_registration_masks_async.assert_called_once_with(
            output_path=Path("/fake/path"), slice_selection=SliceSelection.CURRENT_SLICE
        )


def test_on_export_registration_masks_clicked_current_subject(analysis_widget):
    with patch(
        "napari_brainways.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "output_path": Path("/fake/path"),
            "slice_selection": "Current Subject",
        }
        analysis_widget.on_export_registration_masks_clicked()
        analysis_widget.controller.export_registration_masks_async.assert_called_once_with(
            output_path=Path("/fake/path"),
            slice_selection=SliceSelection.CURRENT_SUBJECT,
        )


def test_on_export_registration_masks_clicked_all_subjects(analysis_widget):
    with patch(
        "napari_brainways.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "output_path": Path("/fake/path"),
            "slice_selection": "All Subjects",
        }
        analysis_widget.on_export_registration_masks_clicked()
        analysis_widget.controller.export_registration_masks_async.assert_called_once_with(
            output_path=Path("/fake/path"), slice_selection=SliceSelection.ALL_SUBJECTS
        )
