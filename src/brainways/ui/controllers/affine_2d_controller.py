from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from PyQt5.QtWidgets import QApplication

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.pipeline.brainways_pipeline import PipelineStep
from brainways.ui.controllers.base import Controller
from brainways.ui.widgets.affine_2d_widget import Affine2DWidget

if TYPE_CHECKING:
    from brainways.ui.brainways_ui import BrainwaysUI


class Affine2DController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui=ui)
        self._image: np.ndarray | None = None
        self._atlas_slice: np.ndarray | None = None
        self._params: BrainwaysParams | None = None
        self.input_layer = None
        self.atlas_slice_layer = None
        self.display_scale: float = 1.0
        self.widget = Affine2DWidget(self)
        self._key_bindings = None

    @property
    def name(self) -> str:
        return "Rigid Registration"

    def register_key_bindings(self):
        key_bindings = {
            "Left": (
                self.keybind_modify_params(tx=-1),
                "Move Left",
            ),
            "Right": (
                self.keybind_modify_params(tx=1),
                "Move Right",
            ),
            "Up": (
                self.keybind_modify_params(ty=-1),
                "Move Up",
            ),
            "Down": (
                self.keybind_modify_params(ty=1),
                "Move Down",
            ),
            "Shift-Left": (
                self.keybind_modify_params(tx=-10),
                "Move Left x10",
            ),
            "Shift-Right": (
                self.keybind_modify_params(tx=10),
                "Move Right x10",
            ),
            "Shift-Up": (
                self.keybind_modify_params(ty=-10),
                "Move Up x10",
            ),
            "Shift-Down": (
                self.keybind_modify_params(ty=10),
                "Move Down x10",
            ),
            "Control-Left": (
                self.keybind_modify_params(angle=-1),
                "Rotate Left",
            ),
            "Control-Right": (
                self.keybind_modify_params(angle=1),
                "Rotate Right",
            ),
            "Alt-Left": (
                self.keybind_modify_params(sx=-0.01),
                "Decrease Horizontal Scale",
            ),
            "Alt-Right": (
                self.keybind_modify_params(sx=0.01),
                "Increase Horizontal Scale",
            ),
            "Alt-Up": (
                self.keybind_modify_params(sy=0.01),
                "Increase Horizontal Scale",
            ),
            "Alt-Down": (
                self.keybind_modify_params(sy=-0.01),
                "Decrease Horizontal Scale",
            ),
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
        return params.affine is not None

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return params.atlas is not None

    def default_params(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams:
        return self.run_model(image=image, params=params)

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        affine_params = self.pipeline.find_2d_affine_transform(
            image=image, params=params
        )
        return replace(params, affine=affine_params)

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ):
        self._params = params
        if image is not None:
            self._image = image
            self._atlas_slice = self.pipeline.get_atlas_slice(params).annotation.numpy()
            self.display_scale = min(
                self._image.shape[0] / self._atlas_slice.shape[0],
                self._image.shape[1] / self._atlas_slice.shape[1],
            )
            self.atlas_slice_layer.data = self._atlas_slice
            self.atlas_slice_layer.scale = (self.display_scale, self.display_scale)
            self.widget.set_ranges(
                tx=(-image.shape[1], image.shape[1]),
                ty=(-image.shape[0], image.shape[0]),
            )
            self.input_layer.data = image
            self.ui.update_layer_contrast_limits(self.input_layer)

        if not from_ui:
            self.widget.set_params(
                angle=params.affine.angle,
                tx=params.affine.tx,
                ty=params.affine.ty,
                sx=params.affine.sx,
                sy=params.affine.sy,
            )

        registered_image = self.pipeline.transform_image(
            image=self._image,
            params=params,
            until_step=PipelineStep.AFFINE_2D,
            scale=self.display_scale,
        )

        self.input_layer.data = registered_image

        self.ui.viewer.reset_view()

        # transform = self.pipeline.get_image_to_atlas_transform(
        #     brainways_params=params,
        #     lowres_image_size=self._image.shape,
        #     until_step=PipelineStep.AFFINE_2D,
        # )
        #
        # transformed_atlas_slice = transform.inv().transform_image(
        #     image=self._atlas_slice,
        #     output_size=self._image.shape,
        #     mode="nearest",
        # )
        # self.atlas_slice_layer.data = annotation_outline(transformed_atlas_slice)

    def open(self) -> None:
        if self._is_open:
            return

        self.input_layer = self.ui.viewer.add_image(
            np.zeros((512, 512), np.uint8), name="Input"
        )
        self.input_layer.events.contrast_limits.connect(self.ui.set_contrast_limits)
        self.atlas_slice_layer = self.ui.viewer.add_labels(
            np.zeros((512, 512), np.uint8),
            name="Atlas",
        )
        self.atlas_slice_layer.contour = True
        self.register_key_bindings()

        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return

        self.unregister_key_bindings()
        self.ui.viewer.layers.remove(self.input_layer)
        self.ui.viewer.layers.remove(self.atlas_slice_layer)
        QApplication.instance().processEvents()

        self.input_layer = None
        self.atlas_slice_layer = None

        self._image = None
        self._atlas_slice = None
        self.display_scale = 1.0
        self._params = None
        self._is_open = False

    def keybind_modify_params(
        self,
        angle: Optional[float] = None,
        tx: Optional[float] = None,
        ty: Optional[float] = None,
        sx: Optional[float] = None,
        sy: Optional[float] = None,
    ) -> Callable:
        def _func(_):
            kwargs = {}
            if angle is not None:
                kwargs["angle"] = self._params.affine.angle + angle
            if tx is not None:
                kwargs["tx"] = self._params.affine.tx + tx
            if ty is not None:
                kwargs["ty"] = self._params.affine.ty + ty
            if sx is not None:
                kwargs["sx"] = self._params.affine.sx + sx
            if sy is not None:
                kwargs["sy"] = self._params.affine.sy + sy
            affine_params = replace(self._params.affine, **kwargs)
            params = replace(self._params, affine=affine_params)
            self.show(params)

        return _func

    def on_params_changed(
        self,
        angle: float,
        tx: float,
        ty: float,
        sx: float,
        sy: str,
    ) -> None:
        affine_params = replace(
            self._params.affine, angle=angle, tx=tx, ty=ty, sx=sx, sy=sy
        )
        params = replace(self._params, affine=affine_params)
        self.show(params, from_ui=True)

    def reset_params(self) -> None:
        default_params = self.run_model(self._image, self._params)
        self.show(default_params)

    @property
    def params(self) -> BrainwaysParams:
        return self._params
