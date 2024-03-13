from unittest.mock import Mock, create_autospec

import pytest

from napari_brainways.controllers.registration_controller import RegistrationController
from napari_brainways.widgets.registration_widget import RegistrationView


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
