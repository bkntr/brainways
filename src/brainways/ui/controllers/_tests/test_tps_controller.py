from copy import deepcopy
from dataclasses import replace
from typing import Tuple
from unittest.mock import patch

import numpy as np
import numpy.testing
import pytest
from pytest import fixture

from brainways.pipeline.affine_2d import AffineTransform2DParams
from brainways.pipeline.brainways_params import TPSTransformParams
from brainways.project.info_classes import BrainwaysParams
from brainways.ui.brainways_ui import BrainwaysUI
from brainways.ui.controllers.tps_controller import TpsController
from brainways.ui.utils.test_utils import randomly_modified_params


@fixture
def app_on_tps(opened_app: BrainwaysUI) -> Tuple[BrainwaysUI, TpsController]:
    tps_step_index = [
        isinstance(step, TpsController) for step in opened_app.steps
    ].index(True)
    opened_app.set_step_index_async(tps_step_index)
    controller: TpsController = opened_app.current_step
    return opened_app, controller


@fixture
def elastix_mock(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
):
    app, controller = app_on_tps
    elastix_result = deepcopy(np.array(controller.params.tps.points_dst))
    elastix_result[0, 0] += 1
    with patch(
        "brainways.pipeline.tps.elastix_registration",
        return_value=elastix_result,
    ):
        yield
    print("a")


@pytest.mark.skip
def test_run_elastix(app_on_tps: Tuple[BrainwaysUI, TpsController]):
    app, controller = app_on_tps
    controller.run_elastix_async()


def test_run_elastix_updates_params(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
    elastix_mock,
):
    app, controller = app_on_tps
    expected = np.array(controller.params.tps.points_dst).copy()
    expected[0, 0] -= 1
    controller.run_elastix_async()
    numpy.testing.assert_allclose(controller.params.tps.points_dst, expected, atol=0.1)


def test_run_elastix_updates_ui(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
    elastix_mock,
):
    app, controller = app_on_tps
    expected = controller.params.tps.points_dst.copy()
    expected[0][0] -= 1
    controller.run_elastix_async()
    numpy.testing.assert_allclose(
        controller.points_atlas_layer.data, np.array(expected)[:, ::-1], atol=0.1
    )


def test_run_elastix_keeps_dtype(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
    elastix_mock,
):
    app, controller = app_on_tps
    controller.run_elastix_async()
    assert type(controller._params.tps.points_src) is list
    assert type(controller._params.tps.points_dst) is list


@pytest.mark.parametrize("hemisphere", ["left", "right", "both"])
def test_default_tps_params_uses_hemispheres(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
    hemisphere: str,
):
    app, controller = app_on_tps
    image = controller._image
    params = controller.params
    atlas_params = params.atlas
    atlas_params = replace(atlas_params, hemisphere=hemisphere)
    params = replace(params, atlas=atlas_params)
    default_params = controller.default_params(image, params)

    if hemisphere == "both":
        expected = 24
    else:
        expected = 12

    assert (
        len(default_params.tps.points_src)
        == len(default_params.tps.points_dst)
        == expected
    )


def test_on_points_changed(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
):
    app, controller = app_on_tps
    val = controller.points_atlas_layer.data[0, 0]
    modified = val + 1
    controller.points_atlas_layer.data[0, 0] = modified
    controller.on_points_changed()
    assert controller.params.tps.points_dst[0][1] == modified


def test_on_points_changed_keeps_params_types(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
):
    app, controller = app_on_tps
    val = controller.points_atlas_layer.data[0, 0]
    modified = val + 1
    controller.points_atlas_layer.data[0, 0] = modified
    controller.on_points_changed()
    assert isinstance(controller.params.tps.points_dst, list)
    assert isinstance(controller.params.tps.points_dst[0][0], float)


def test_reset_params(app_on_tps: Tuple[BrainwaysUI, TpsController]):
    app, controller = app_on_tps
    default_tps_params = controller.default_params(controller._image, controller.params)
    controller._params = randomly_modified_params(controller.params)
    assert controller.params.tps != default_tps_params.tps
    controller.reset_params()
    assert controller.params.tps == default_tps_params.tps


def test_previous_params(app_on_tps: Tuple[BrainwaysUI, TpsController]):
    app, controller = app_on_tps
    previous_params = controller.params
    next_params = replace(
        previous_params,
        tps=TPSTransformParams(np.random.rand(10, 2), np.random.rand(10, 2)),
    )
    controller.show(next_params, from_ui=True)
    assert controller.params == next_params
    controller.previous_params()
    assert controller.params == previous_params


def test_previous_next_params(app_on_tps: Tuple[BrainwaysUI, TpsController]):
    app, controller = app_on_tps
    next_params = replace(
        controller.params,
        tps=TPSTransformParams(np.random.rand(10, 2), np.random.rand(10, 2)),
    )
    controller.show(next_params, from_ui=True)
    controller.previous_params()
    controller.next_params()
    assert controller.params == next_params


def test_previous_next_previous_params(app_on_tps: Tuple[BrainwaysUI, TpsController]):
    app, controller = app_on_tps
    previous_params = controller.params
    next_params = replace(
        previous_params,
        tps=TPSTransformParams(np.random.rand(10, 2), np.random.rand(10, 2)),
    )
    controller.show(next_params, from_ui=True)
    controller.previous_params()
    controller.next_params()
    controller.previous_params()
    assert controller.params == previous_params


def test_previous_params_empty_does_nothing(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
):
    app, controller = app_on_tps
    previous_params = controller.params
    controller.previous_params()
    assert controller.params == previous_params


def test_next_params_empty_does_nothing(app_on_tps: Tuple[BrainwaysUI, TpsController]):
    app, controller = app_on_tps
    previous_params = controller.params
    controller.next_params()
    assert controller.params == previous_params


def test_change_params_empties_next_params(
    app_on_tps: Tuple[BrainwaysUI, TpsController],
):
    app, controller = app_on_tps
    next_params1 = replace(
        controller.params,
        tps=TPSTransformParams(np.random.rand(10, 2), np.random.rand(10, 2)),
    )
    next_params2 = replace(
        controller.params,
        tps=TPSTransformParams(np.random.rand(10, 2), np.random.rand(10, 2)),
    )
    controller.show(next_params1, from_ui=True)
    controller.previous_params()
    controller.show(next_params2, from_ui=True)
    controller.next_params()
    assert controller.params == next_params2


def test_enabled_false_by_default():
    params = BrainwaysParams()
    enabled = TpsController.enabled(params)
    assert enabled is False


def test_enabled_with_affine_params():
    params = BrainwaysParams(affine=AffineTransform2DParams())
    enabled = TpsController.enabled(params)
    assert enabled is True
