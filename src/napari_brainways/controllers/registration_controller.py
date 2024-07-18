from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

import napari.layers
import numpy as np

from brainways.pipeline.brainways_params import AtlasRegistrationParams, BrainwaysParams
from brainways.utils.image import brain_mask, nonzero_bounding_box
from napari_brainways.controllers.base import Controller
from napari_brainways.utils.general_utils import update_layer_contrast_limits
from napari_brainways.widgets.registration_widget import RegistrationView

if TYPE_CHECKING:
    from napari_brainways.brainways_ui import BrainwaysUI


class RegistrationController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui=ui)
        self.widget = RegistrationView(self)
        self.input_layer: napari.layers.Image | None = None
        self.mask_layer: napari.layers.Image | None = None
        self.atlas_slice_layer: napari.layers.Image | None = None
        self._image: np.ndarray | None = None
        self._params: BrainwaysParams | None = None
        self._input_box = None
        self._key_bindings = None

    @property
    def name(self) -> str:
        return "Atlas Registration"

    def register_key_bindings(self):
        key_bindings = {
            "Left": (
                self.get_keybind_fn(self.widget.modify_ap, value=-1),
                "Decrease AP",
            ),
            "Right": (
                self.get_keybind_fn(self.widget.modify_ap, value=1),
                "Increase AP",
            ),
            "Shift-Left": (
                self.get_keybind_fn(self.widget.modify_ap, value=-10),
                "Decrease AP -10",
            ),
            "Shift-Right": (
                self.get_keybind_fn(self.widget.modify_ap, value=10),
                "Increase AP +10",
            ),
            "Control-Left": (
                self.get_keybind_fn(self.widget.modify_hemisphere, value="left"),
                "Left Hemisphere",
            ),
            "Control-Right": (
                self.get_keybind_fn(self.widget.modify_hemisphere, value="right"),
                "Right Hemisphere",
            ),
            "Control-Down": (
                self.get_keybind_fn(self.widget.modify_hemisphere, value="both"),
                "Both Hemispheres",
            ),
            "Control-Up": (
                self.get_keybind_fn(self.widget.modify_hemisphere, value="both"),
                "Both Hemispheres",
            ),
            "Alt-Left": (
                self.get_keybind_fn(self.widget.modify_rot_horizontal, value=-1),
                "Horizontal Rotation Left",
            ),
            "Alt-Right": (
                self.get_keybind_fn(self.widget.modify_rot_horizontal, value=1),
                "Horizontal Rotation Right",
            ),
            "Alt-Up": (
                self.get_keybind_fn(self.widget.modify_rot_sagittal, value=-1),
                "Sagittal Rotation Up",
            ),
            "Alt-Down": (
                self.get_keybind_fn(self.widget.modify_rot_sagittal, value=1),
                "Sagittal Rotation Down",
            ),
            "?": (
                self.show_help,
                "Show Help",
            ),
        }
        for key, (func, _) in key_bindings.items():
            self.ui.viewer.bind_key(key, func, overwrite=True)

        self._key_bindings = key_bindings

    def get_keybind_fn(self, function: Callable, value: Any) -> Callable:
        def _fn(_):
            function(value=value)

        return _fn

    def show_help(self, _=None):
        self.widget.show_help(key_bindings=self._key_bindings)

    def unregister_key_bindings(self):
        for key in self._key_bindings:
            if key in self.ui.viewer.keymap:
                self.ui.viewer.keymap.pop(key)
        self._key_bindings = None

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return params.atlas is not None

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return True

    def pipeline_loaded(self):
        self.widget.update_model(ap_min=0, ap_max=self.pipeline.atlas.shape[0] - 1)

    def default_params(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams:
        if self.model_available():
            return self.run_model(image=image, params=params)
        else:
            return replace(
                params,
                atlas=AtlasRegistrationParams(ap=self.pipeline.atlas.shape[0] // 2),
            )

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        model_registration_params = (
            self.pipeline.atlas_registration.run_automatic_registration(image)
        )
        atlas_params = AtlasRegistrationParams(
            ap=model_registration_params.ap,
            rot_frontal=model_registration_params.rot_frontal,
            rot_horizontal=model_registration_params.rot_horizontal,
            rot_sagittal=model_registration_params.rot_sagittal,
            hemisphere=model_registration_params.hemisphere,
            confidence=model_registration_params.confidence,
        )
        return replace(params, atlas=atlas_params)

    def on_run_model_button_click(self):
        if self.model_available():
            params = self.run_model(image=self._image, params=self._params)
            self.show(params)

    def model_available(self) -> bool:
        return self.pipeline.atlas_registration.automatic_registration_available()

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        self._params = params

        if image is not None:
            self._image = image
            mask = brain_mask(image)
            self._input_box = nonzero_bounding_box(mask)
            self.mask_layer.data = mask
            self.input_layer.data = image
            update_layer_contrast_limits(self.input_layer, (0.01, 0.98))

        if not from_ui:
            self.widget.set_registration_params(
                ap=params.atlas.ap,
                rot_horizontal=params.atlas.rot_horizontal,
                rot_sagittal=params.atlas.rot_sagittal,
                hemisphere=params.atlas.hemisphere,
            )

        atlas_slice = self.pipeline.get_atlas_slice(self.params).reference.numpy()
        self.atlas_slice_layer.data = atlas_slice
        update_layer_contrast_limits(self.atlas_slice_layer, (0.01, 0.98))

        atlas_box = self.pipeline.atlas.bounding_box(int(self.params.atlas.ap))
        input_scale = atlas_box[3] / self._input_box[3]
        self.input_layer.scale = (input_scale, input_scale)
        self.mask_layer.scale = (input_scale, input_scale)

        tx = (self._input_box[0] + self._input_box[0] * 0.1) * input_scale
        ty = self._input_box[1] * input_scale - atlas_box[1]
        self.atlas_slice_layer.translate = (ty, tx)

        if not from_ui:
            self.ui.viewer.reset_view()

    def on_params_changed(
        self,
        ap: float,
        rot_horizontal: float,
        rot_sagittal: float,
        hemisphere: str,
    ):
        params = replace(
            self._params,
            atlas=AtlasRegistrationParams(
                ap=ap,
                rot_horizontal=rot_horizontal,
                rot_sagittal=rot_sagittal,
                hemisphere=hemisphere,
            ),
        )
        self.show(params, from_ui=True)

    def open(self):
        if self._is_open:
            return

        self.input_layer = self.ui.viewer.add_image(
            np.zeros((512, 512), np.uint8),
            name="Input",
        )
        self.mask_layer = self.ui.viewer.add_image(
            np.zeros((512, 512), np.uint8),
            name="Mask",
            visible=False,
            colormap="green",
            blending="additive",
        )
        self.atlas_slice_layer = self.ui.viewer.add_image(
            np.zeros(
                (self.pipeline.atlas.shape[1], self.pipeline.atlas.shape[2]),
                np.uint8,
            ),
            name="Atlas Slice",
        )
        self.input_layer.translate = (0, self.pipeline.atlas.shape[2])
        self.mask_layer.translate = (0, self.pipeline.atlas.shape[2])
        self.register_key_bindings()

        self._is_open = True

    def close(self):
        if not self._is_open:
            return

        self.unregister_key_bindings()
        self.ui.viewer.layers.remove(self.input_layer)
        self.ui.viewer.layers.remove(self.mask_layer)
        self.ui.viewer.layers.remove(self.atlas_slice_layer)
        self.input_layer = None
        self.mask_layer = None
        self.atlas_slice_layer = None
        self._params = None
        self._is_open = False

    @property
    def params(self) -> BrainwaysParams:
        return self._params
