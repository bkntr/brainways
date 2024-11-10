# mypy: disable-error-code="method-assign"

from unittest.mock import MagicMock, Mock, create_autospec

import pytest
from qtpy.QtWidgets import QMessageBox

from brainways.ui.controllers.registration_controller import RegistrationController
from brainways.ui.widgets.registration_widget import RegistrationView


@pytest.mark.parametrize("model_available", argvalues=[False, True])
def test_has_registration_model(model_available: bool):
    MockController = create_autospec(RegistrationController)
    mock_controller = MockController(ui=Mock())
    mock_controller.model_available.return_value = model_available
    widget = RegistrationView(mock_controller)
    widget.update_model(ap_min=0, ap_max=1)
    if model_available:
        assert widget.run_model_button.toolTip() == ""
    else:
        assert len(widget.run_model_button.toolTip()) > 0


def test_confirm_apply_rotation(monkeypatch):
    widget = RegistrationView(MagicMock())
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    widget.confirm_apply_rotation()
    widget.controller.on_apply_rotation_to_subject_click.assert_called_once()


def test_no_confirm_apply_rotation(monkeypatch):
    widget = RegistrationView(MagicMock())
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    widget.confirm_apply_rotation()
    widget.controller.on_apply_rotation_to_subject_click.assert_not_called()


def test_confirm_evenly_space_slices(monkeypatch):
    widget = RegistrationView(MagicMock())
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    widget.confirm_evenly_space_slices()
    widget.controller.on_evenly_space_slices_on_ap_axis_click.assert_called_once()


def test_no_confirm_evenly_space_slices(monkeypatch):
    widget = RegistrationView(MagicMock())
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    widget.confirm_evenly_space_slices()
    widget.controller.on_evenly_space_slices_on_ap_axis_click.assert_not_called()
