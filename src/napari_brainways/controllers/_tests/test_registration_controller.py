from brainways.project.info_classes import BrainwaysParams
from napari_brainways.controllers.registration_controller import RegistrationController


def test_enabled():
    params = BrainwaysParams()
    enabled = RegistrationController.enabled(params)
    assert enabled is True
