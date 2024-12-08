from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from brainways.project.info_classes import (
    MaskFileFormat,
    RegisteredPixelValues,
    SliceSelection,
)
from brainways.ui.widgets.analysis_widget import AnalysisWidget


@pytest.fixture
def mock_controller():
    controller = Mock()
    controller.ui.prompt_user_slices_have_missing_params = Mock(return_value=True)
    return controller


@pytest.fixture
def analysis_widget(mock_controller) -> AnalysisWidget:
    return AnalysisWidget(mock_controller)


def test_on_export_registration_masks_clicked_current_slice(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "pixel_value_mode": "Micron Coordinates",
            "output_path": Path("/fake/path"),
            "slice_selection": "Current Slice",
            "file_format": "csv",
        }
        analysis_widget.on_export_registration_masks_clicked()
        analysis_widget.controller.export_registration_masks_async.assert_called_once_with(
            output_path=Path("/fake/path"),
            pixel_value_mode=RegisteredPixelValues.MICRON_COORDINATES,
            slice_selection=SliceSelection.CURRENT_SLICE,
            file_format=MaskFileFormat.CSV,
        )


def test_on_export_registration_masks_clicked_current_subject(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "pixel_value_mode": "Pixel Coordinates",
            "output_path": Path("/fake/path"),
            "slice_selection": "Current Subject",
            "file_format": "csv",
        }
        analysis_widget.on_export_registration_masks_clicked()
        analysis_widget.controller.export_registration_masks_async.assert_called_once_with(
            output_path=Path("/fake/path"),
            pixel_value_mode=RegisteredPixelValues.PIXEL_COORDINATES,
            slice_selection=SliceSelection.CURRENT_SUBJECT,
            file_format=MaskFileFormat.CSV,
        )


def test_on_export_registration_masks_clicked_all_subjects(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "pixel_value_mode": "Structure IDs",
            "output_path": Path("/fake/path"),
            "slice_selection": "All Subjects",
            "file_format": "csv",
        }
        analysis_widget.on_export_registration_masks_clicked()
        analysis_widget.controller.export_registration_masks_async.assert_called_once_with(
            output_path=Path("/fake/path"),
            pixel_value_mode=RegisteredPixelValues.STRUCTURE_IDS,
            slice_selection=SliceSelection.ALL_SUBJECTS,
            file_format=MaskFileFormat.CSV,
        )


def test_on_export_registration_masks_clicked_different_format(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "pixel_value_mode": "Structure IDs",
            "output_path": Path("/fake/path"),
            "slice_selection": "Current Slice",
            "file_format": "npz",
        }
        analysis_widget.on_export_registration_masks_clicked()
        analysis_widget.controller.export_registration_masks_async.assert_called_once_with(
            output_path=Path("/fake/path"),
            pixel_value_mode=RegisteredPixelValues.STRUCTURE_IDS,
            slice_selection=SliceSelection.CURRENT_SLICE,
            file_format=MaskFileFormat.NPZ,
        )


def test_on_export_registration_masks_clicked_no_missing_params(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values, patch.object(
        analysis_widget.controller.ui,
        "prompt_user_slices_have_missing_params",
        return_value=False,
    ):
        analysis_widget.on_export_registration_masks_clicked()

        # Ensure request_values is not called
        mock_request_values.assert_not_called()

        # Ensure export_registration_masks_async is not called
        analysis_widget.controller.export_registration_masks_async.assert_not_called()


def test_on_export_slice_locations_clicked_current_slice(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = {
            "output_path": Path("/fake/path"),
            "slice_selection": "Current Slice",
        }
        analysis_widget.on_export_slice_locations_clicked()
        analysis_widget.controller.export_slice_locations.assert_called_once_with(
            output_path=Path("/fake/path.csv"),
            slice_selection=SliceSelection.CURRENT_SLICE,
        )


def test_on_export_slice_locations_clicked_no_missing_params(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values, patch.object(
        analysis_widget.controller.ui,
        "prompt_user_slices_have_missing_params",
        return_value=False,
    ):
        analysis_widget.on_export_slice_locations_clicked()

        # Ensure request_values is not called
        mock_request_values.assert_not_called()

        # Ensure export_slice_locations is not called
        analysis_widget.controller.export_slice_locations.assert_not_called()


def test_on_export_slice_locations_clicked_request_values_none(analysis_widget):
    with patch(
        "brainways.ui.widgets.analysis_widget.request_values"
    ) as mock_request_values:
        mock_request_values.return_value = None
        analysis_widget.on_export_slice_locations_clicked()

        # Ensure export_slice_locations is not called
        analysis_widget.controller.export_slice_locations.assert_not_called()
