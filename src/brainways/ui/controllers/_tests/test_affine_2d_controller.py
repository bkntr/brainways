from dataclasses import replace
from typing import Tuple

from pytest import fixture

from brainways.pipeline.atlas_registration import AtlasRegistrationParams
from brainways.project.info_classes import BrainwaysParams
from brainways.ui.brainways_ui import BrainwaysUI
from brainways.ui.controllers.affine_2d_controller import Affine2DController
from brainways.utils.test_utils import randomly_modified_params


@fixture
def app_on_affine_2d(opened_app: BrainwaysUI) -> Tuple[BrainwaysUI, Affine2DController]:
    affine_2d_step_index = [
        isinstance(step, Affine2DController) for step in opened_app.steps
    ].index(True)
    opened_app.set_step_index_async(affine_2d_step_index)
    controller: Affine2DController = opened_app.current_step
    params = controller.params
    params = replace(params, affine=replace(params.affine, angle=10))
    controller.show(params)
    return opened_app, controller


def test_reset_params(app_on_affine_2d: Tuple[BrainwaysUI, Affine2DController]):
    app, controller = app_on_affine_2d
    assert controller.params.affine.angle == 10
    controller.reset_params()
    assert controller.params.affine.angle == 0


def test_reset_params_updates_ui(
    app_on_affine_2d: Tuple[BrainwaysUI, Affine2DController]
):
    app, controller = app_on_affine_2d
    assert controller.widget.params_widget.angle.value == 10
    controller.reset_params()
    assert controller.widget.params_widget.angle.value == 0


def test_update_params(app_on_affine_2d: Tuple[BrainwaysUI, Affine2DController]):
    app, controller = app_on_affine_2d
    params = controller.params
    modified_params = randomly_modified_params(controller.params)
    assert params != modified_params
    controller.show(modified_params)
    assert controller.params == modified_params


def test_on_params_changed(app_on_affine_2d: Tuple[BrainwaysUI, Affine2DController]):
    app, controller = app_on_affine_2d
    params = controller.params
    modified_affine_params = randomly_modified_params(controller.params).affine
    modified_params = replace(params, affine=modified_affine_params)
    assert params != modified_params
    controller.on_params_changed(
        angle=modified_affine_params.angle,
        tx=modified_affine_params.tx,
        ty=modified_affine_params.ty,
        sx=modified_affine_params.sx,
        sy=modified_affine_params.sy,
    )
    assert controller.params == modified_params


def test_enabled_false_by_default():
    params = BrainwaysParams()
    enabled = Affine2DController.enabled(params)
    assert enabled is False


def test_enabled_when_has_atlas():
    params = BrainwaysParams(atlas=AtlasRegistrationParams())
    enabled = Affine2DController.enabled(params)
    assert enabled is True
