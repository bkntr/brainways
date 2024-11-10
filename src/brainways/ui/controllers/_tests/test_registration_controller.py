# mypy: disable-error-code="method-assign"

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import BrainwaysParams, SubjectInfo
from brainways.ui.controllers.registration_controller import RegistrationController


def test_enabled():
    params = BrainwaysParams()
    enabled = RegistrationController.enabled(params)
    assert enabled is True


@pytest.fixture
def setup_controller() -> RegistrationController:
    mock_ui = MagicMock()
    controller = RegistrationController(mock_ui)
    controller.ui.current_subject = BrainwaysSubject(
        subject_info=SubjectInfo("mock"), slice_infos=[], project=MagicMock()
    )
    return controller


def test_apply_rotation_to_subject(setup_controller: RegistrationController):
    controller = setup_controller

    # Set the rotation parameters
    controller._params = BrainwaysParams(
        atlas=AtlasRegistrationParams(rot_horizontal=10, rot_sagittal=20)
    )

    # Apply rotation to all slices
    controller.on_apply_rotation_to_subject_click()

    # Get the default parameters
    updated_params = controller.default_params(
        image=MagicMock(), params=BrainwaysParams()
    )

    # Check that the rotation has been applied
    assert updated_params.atlas is not None
    assert updated_params.atlas.rot_horizontal == 10
    assert updated_params.atlas.rot_sagittal == 20


def test_default_params_model_available(setup_controller: RegistrationController):
    controller = setup_controller
    controller.model_available = MagicMock(return_value=True)
    mock_params = BrainwaysParams(
        atlas=AtlasRegistrationParams(ap=50, rot_horizontal=10, rot_sagittal=20)
    )
    controller.run_model = MagicMock(return_value=mock_params)

    default_params = controller.default_params(
        image=MagicMock(), params=BrainwaysParams()
    )

    assert default_params == mock_params


def test_default_params_model_not_available(setup_controller: RegistrationController):
    controller = setup_controller
    controller.model_available = MagicMock(return_value=False)
    controller.ui.project.pipeline.atlas.shape = [100, 100, 100]  # type: ignore

    default_params = controller.default_params(
        image=MagicMock(), params=BrainwaysParams()
    )

    assert default_params.atlas == AtlasRegistrationParams(ap=50)  # Middle of the atlas


def test_default_params_with_subject_rotation(setup_controller: RegistrationController):
    controller = setup_controller
    controller.model_available = MagicMock(return_value=False)
    controller.ui.project.pipeline.atlas.shape = [100, 100, 100]  # type: ignore
    subject_info = controller.ui.current_subject.subject_info
    controller.ui.current_subject.subject_info = replace(
        subject_info, rotation=(10, 20)
    )

    default_params = controller.default_params(
        image=MagicMock(), params=BrainwaysParams()
    )

    assert default_params.atlas == AtlasRegistrationParams(
        ap=50, rot_horizontal=10, rot_sagittal=20
    )


def test_run_model_with_available_model(setup_controller: RegistrationController):
    controller = setup_controller
    mock_registration_params = AtlasRegistrationParams(
        ap=50,
        rot_frontal=5,
        rot_horizontal=10,
        rot_sagittal=15,
        hemisphere="left",
        confidence=0.9,
    )

    controller.pipeline.atlas_registration.run_automatic_registration = MagicMock(
        return_value=mock_registration_params
    )

    result = controller.run_model(image=MagicMock(), params=BrainwaysParams())

    assert result.atlas == mock_registration_params


def test_run_model_with_subject_rotation(setup_controller: RegistrationController):
    controller = setup_controller
    mock_registration_params = AtlasRegistrationParams(
        ap=50,
        rot_frontal=5,
        rot_horizontal=10,
        rot_sagittal=15,
        hemisphere="left",
        confidence=0.9,
    )

    controller.pipeline.atlas_registration.run_automatic_registration = MagicMock(
        return_value=mock_registration_params
    )
    subject_info = controller.ui.current_subject.subject_info
    controller.ui.current_subject.subject_info = replace(
        subject_info, rotation=(20, 30)
    )

    result = controller.run_model(image=MagicMock(), params=BrainwaysParams())

    expected_params = AtlasRegistrationParams(
        ap=50,
        rot_frontal=5,
        rot_horizontal=20,  # Overridden by subject rotation
        rot_sagittal=30,  # Overridden by subject rotation
        hemisphere="left",
        confidence=0.9,
    )

    assert result.atlas == expected_params


def test_on_evenly_space_slices_on_ap_axis_click(
    setup_controller: RegistrationController,
):
    controller = setup_controller
    controller.show = MagicMock()
    controller.ui.current_subject.evenly_space_slices_on_ap_axis = MagicMock()
    params = BrainwaysParams(atlas=AtlasRegistrationParams(ap=1))
    controller.ui.current_params = params
    controller.on_evenly_space_slices_on_ap_axis_click()
    controller.ui.persist_current_params.assert_called_once()  # type: ignore
    controller.ui.current_subject.evenly_space_slices_on_ap_axis.assert_called_once()
    controller.show.assert_called_once_with(params)  # type: ignore
    controller.ui.save_subject.assert_called_once()  # type: ignore
