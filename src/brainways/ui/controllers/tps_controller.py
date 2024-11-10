from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, List

import napari.layers
import numpy as np
from napari.qt.threading import FunctionWorker
from PyQt5.QtWidgets import QApplication

from brainways.pipeline.brainways_params import BrainwaysParams, TPSTransformParams
from brainways.pipeline.brainways_pipeline import PipelineStep
from brainways.transforms.tps_transform import TPSTransform
from brainways.ui.controllers.base import Controller
from brainways.ui.widgets.tps_widget import TpsWidget
from brainways.utils.image import brain_mask, nonzero_bounding_box

if TYPE_CHECKING:
    from brainways.ui.brainways_ui import BrainwaysUI


class TpsController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui=ui)
        self._params: BrainwaysParams | None = None
        self._image: np.ndarray | None = None
        self.input_layer: napari.layers.Image | None = None
        self.atlas_layer: napari.layers.Labels | None = None
        self.points_input_layer: napari.layers.Points | None = None
        self.points_atlas_layer: napari.layers.Points | None = None
        self.widget = TpsWidget(self)
        self._key_bindings = None
        self._prev_params: List[BrainwaysParams] | None = None
        self._next_params: List[BrainwaysParams] | None = None

    @property
    def name(self) -> str:
        return "Non-Rigid Registration"

    def register_key_bindings(self):
        key_bindings = {
            "Control-Z": (self.previous_params, "Set Previous Points"),
            "Control-Y": (self.next_params, "Set Next Points"),
            "Control-Shift-Z": (self.next_params, "Set Next Points"),
            "A": (self.set_points_mode_add, "Add Point"),
            "?": (self.show_help, "Show Help"),
        }
        for key, (func, _) in key_bindings.items():
            self.ui.viewer.bind_key(key, func, overwrite=True)

        self._key_bindings = key_bindings

    def unregister_key_bindings(self):
        for key in self._key_bindings:
            if key in self.ui.viewer.keymap:
                self.ui.viewer.keymap.pop(key)
        self._key_bindings = None

    def show_help(self, _=None):
        self.widget.show_help(key_bindings=self._key_bindings)

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return params.tps is not None

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return params.affine is not None

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        affine_image = self.pipeline.transform_image(
            image=image, params=params, until_step=PipelineStep.AFFINE_2D
        )
        atlas_slice = self.pipeline.get_atlas_slice(params)
        return self.pipeline.tps.find_registration_params(
            image=affine_image, atlas_slice=atlas_slice, params=params
        )

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        assert params.tps is not None

        if from_ui:
            self._prev_params.append(self._params)
            self._next_params = []

        self._params = params
        display_scale = 1 / min(params.affine.sx, params.affine.sy)

        if image is not None:
            self._image = image
            self._next_params = []
            self._prev_params = []

            atlas_slice = self.pipeline.get_atlas_slice(params)
            self.atlas_layer.data = atlas_slice.annotation.numpy()
            self.atlas_layer.scale = (display_scale, display_scale)
            self.points_input_layer.scale = (display_scale, display_scale)
            self.points_atlas_layer.scale = (display_scale, display_scale)
        with self.points_input_layer.events.data.blocker():
            np_pts = np.array(params.tps.points_src)[:, ::-1]
            self.points_input_layer.data = np_pts.copy()
            self.points_input_layer.selected_data = set()
        with self.points_atlas_layer.events.data.blocker():
            np_pts = np.array(params.tps.points_dst)[:, ::-1]
            self.points_atlas_layer.data = np_pts.copy()
            self.points_atlas_layer.selected_data = set()

        self.input_layer.data = self.pipeline.transform_image(
            image=self._image,
            params=params,
            until_step=PipelineStep.TPS,
            scale=display_scale,
        )

        if image is not None:
            self.ui.update_layer_contrast_limits(self.input_layer)
            self.ui.viewer.reset_view()

    def open(self) -> None:
        if self._is_open:
            return

        self.input_layer = self.ui.viewer.add_image(
            np.zeros((10, 10), np.uint8),
            name="Input",
        )
        self.input_layer.events.contrast_limits.connect(self.ui.set_contrast_limits)
        self.atlas_layer = self.ui.viewer.add_labels(
            np.zeros((10, 10), np.uint8),
            name="Atlas",
        )
        self.atlas_layer.contour = True
        self.points_input_layer = self.ui.viewer.add_points(
            name="Input Points",
            face_color="green",
            border_color="#00ff0064",
            size=5,
            border_width=0.5,
            visible=False,
        )
        self.points_atlas_layer = self.ui.viewer.add_points(
            name="Atlas Points",
            face_color="blue",
            border_color="#0000ff64",
            border_width=0.8,
            size=5,
        )
        self.points_atlas_layer.mode = "select"
        self.points_atlas_layer.events.data.connect(self.on_points_changed)

        self.points_atlas_layer.bind_key("a", self.set_points_mode_add, overwrite=True)
        self.points_atlas_layer.bind_key(
            "s", self.set_points_mode_select, overwrite=True
        )
        self._prev_params = []
        self._next_params = []

        self.register_key_bindings()
        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return

        self.points_atlas_layer.events.data.disconnect(self.on_points_changed)
        self.ui.viewer.layers.remove(self.input_layer)
        self.ui.viewer.layers.remove(self.atlas_layer)
        self.ui.viewer.layers.remove(self.points_input_layer)
        self.ui.viewer.layers.remove(self.points_atlas_layer)
        QApplication.instance().processEvents()
        self._image = None
        self._params = None
        self.input_layer = None
        self.atlas_layer = None
        self.points_input_layer = None
        self.points_atlas_layer = None
        self._prev_params = None
        self._next_params = None
        self.unregister_key_bindings()
        self._is_open = False

    def default_params(self, image: np.ndarray, params: BrainwaysParams):
        is_both_hemispheres = params.atlas.hemisphere == "both"
        nx = 6 if is_both_hemispheres else 3
        ny = 4

        dst = self.pipeline.get_atlas_slice(params).reference.numpy()
        x, y, w, h = nonzero_bounding_box(brain_mask(dst))
        xs, ys = np.meshgrid(np.linspace(x, x + w, nx), np.linspace(y, y + h, ny))
        points = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)

        return replace(
            params,
            tps=TPSTransformParams(points.copy().tolist(), points.copy().tolist()),
        )

    def set_points_mode_add(self, _=None):
        self.ui.viewer.layers.selection = {self.points_atlas_layer}
        self.points_atlas_layer.mode = "add"

    def set_points_mode_select(self, _=None):
        self.ui.viewer.layers.selection = {self.points_atlas_layer}
        self.points_atlas_layer.mode = "select"

    def on_points_changed(self, event=None):
        if event is not None and event.action not in ("added", "changed"):
            return

        if self.points_atlas_layer.mode == "add":
            point_to_add = self.points_atlas_layer.data[-1]
            transform = TPSTransform(params=self._params.tps).inv()
            point_to_add_transformed = transform.transform_points(
                point_to_add[None, ::-1]
            )
            self.points_input_layer.add(point_to_add_transformed[0, ::-1])
            self.points_atlas_layer.mode = "select"

        points_src = self.points_input_layer.data[:, ::-1]
        points_dst = self.points_atlas_layer.data[:, ::-1]

        tps_params = TPSTransformParams(
            points_src=points_src.astype(np.float32).tolist(),
            points_dst=points_dst.astype(np.float32).tolist(),
        )
        updated_params = replace(self._params, tps=tps_params)
        self.show(params=updated_params, from_ui=True)

    def _run_elastix(self) -> BrainwaysParams:
        return self.run_model(self._image, self._params)

    def _run_elastix_returned(self, params: BrainwaysParams):
        self.show(params, from_ui=True)

    def run_elastix_async(self) -> FunctionWorker:
        return self.ui.do_work_async(
            self._run_elastix,
            return_callback=self._run_elastix_returned,
            progress_label="Running Elastix...",
        )

    def reset_params(self):
        self.show(params=self.default_params(self._image, self._params), from_ui=True)

    def previous_params(self, _=None):
        if len(self._prev_params) == 0:
            return
        self._next_params.append(self._params)
        self.show(self._prev_params.pop())

    def next_params(self, _=None):
        if len(self._next_params) == 0:
            return
        self._prev_params.append(self._params)
        self.show(self._next_params.pop())

    @property
    def params(self) -> BrainwaysParams:
        return self._params
