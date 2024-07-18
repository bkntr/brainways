import random
from pathlib import Path
from typing import Optional

import click
import napari
import numpy as np
import PIL.Image
import torch
from brainglobe_atlasapi import BrainGlobeAtlas
from magicgui import magicgui

from brainways_reg_model.model.model import BrainwaysRegModel
from brainways_reg_model.utils.image import slice_contrast_values
from brainways_reg_model.utils.paths import REAL_DATA_ROOT, REAL_TRAINED_MODEL_ROOT
from brainways_reg_model.utils.slice_atlas import slice_atlas


class RegistrationAnnotator:
    def __init__(
        self,
        images_root: str,
        filter: Optional[str] = None,
    ):
        self.image_paths = [
            path for path in Path(images_root).rglob(filter or "*") if path.is_file()
        ]
        self.current_image_idx = 0

        self.viewer = napari.Viewer()
        self.model = BrainwaysRegModel.load_from_checkpoint(
            REAL_TRAINED_MODEL_ROOT / "model.ckpt"
        )
        self.model.freeze()
        self.atlas = BrainGlobeAtlas(self.model.config.data.atlas.name)
        self.atlas_volume = torch.as_tensor(self.atlas.reference.astype(np.float32))

        self._input_translate = (0, self.atlas.shape[2])
        self._overlay_translate = (
            self.atlas.shape[1],
            self.atlas.shape[2] // 2,
        )

        self.input_layer = self.viewer.add_image(
            np.zeros((512, 512), np.uint8),
            name="Input",
        )
        self.atlas_slice_layer = self.viewer.add_image(
            np.zeros((self.atlas.shape[1], self.atlas.shape[2]), np.uint8),
            name="Atlas Slice",
        )
        self.input_layer.translate = self._input_translate

        self.registration_params_widget = magicgui(
            self.registration_params,
            auto_call=True,
            ap={
                "label": "Anterior-Posterior",
                "widget_type": "FloatSlider",
                "max": self.atlas.shape[0] - 1,
                "enabled": False,
            },
            rot_frontal={
                "label": "Frontal Rotation",
                "widget_type": "FloatSlider",
                "min": -30,
                "max": 30,
                "enabled": False,
            },
            rot_horizontal={
                "label": "Horizontal Rotation",
                "widget_type": "FloatSlider",
                "min": -15,
                "max": 15,
                "enabled": False,
            },
            hemisphere={
                "label": "Hemisphere",
                "widget_type": "RadioButtons",
                "orientation": "horizontal",
                "choices": [("Left", "left"), ("Both", "both"), ("Right", "right")],
            },
            confidence={
                "label": "Confidence",
                "widget_type": "FloatSlider",
                "min": 0,
                "max": 1,
                "step": 0.01,
            },
        )

        self.image_slider_widget = magicgui(
            self.image_slider,
            auto_call=True,
            image_number={
                "widget_type": "Slider",
                "label": "Image #",
                "min": 1,
                "max": len(self.image_paths),
            },
        )

        self.viewer.window.add_dock_widget(
            self.registration_params_widget, name="Annotate", area="right"
        )

        self.viewer.window.add_dock_widget(
            self.image_slider_widget, name="Images", area="right"
        )

        self.change_image()

        self.viewer.bind_key("r", self.random_image)

    def random_image(self, viewer: napari.Viewer):
        self.current_image_idx = random.randint(0, len(self.image_paths) - 1)
        self.change_image()

    def registration_params(
        self,
        ap: float,
        rot_frontal: float,
        rot_horizontal: float,
        hemisphere: str,
        confidence: float,
    ) -> napari.types.LayerDataTuple:
        pass

    def image_slider(self, image_number: int):
        image_idx = image_number - 1
        if self.current_image_idx != image_idx:
            self.current_image_idx = image_idx
            self.change_image()

    @property
    def image_path(self) -> str:
        return self.image_paths[self.current_image_idx].as_posix()

    def change_image(self):
        image = PIL.Image.open(self.image_path)
        image = torch.as_tensor(np.array(image, dtype=np.float32))[None, ...]
        params = self.model.predict(image)

        self.registration_params_widget(
            ap=params.ap,
            rot_frontal=params.rot_frontal,
            rot_horizontal=params.rot_horizontal,
            hemisphere=params.hemisphere,
            confidence=params.confidence,
            update_widget=True,
        )
        atlas_slice = slice_atlas(
            shape=self.atlas_volume.shape[1:],
            volume=self.atlas_volume,
            ap=params.ap,
            si=self.atlas_volume.shape[1] / 2,
            lr=self.atlas_volume.shape[2] / 2,
            rot_frontal=params.rot_frontal,
            rot_horizontal=params.rot_horizontal,
            rot_sagittal=0,
        ).numpy()

        self.input_layer.data = np.array(image)
        self.input_layer.reset_contrast_limits_range()
        self.input_layer.contrast_limits = slice_contrast_values(self.input_layer.data)
        self.atlas_slice_layer.data = atlas_slice
        self.atlas_slice_layer.reset_contrast_limits_range()
        self.atlas_slice_layer.reset_contrast_limits()
        input_scale = min(
            atlas_slice.shape[0] / image.shape[1], atlas_slice.shape[1] / image.shape[2]
        )
        self.input_layer.scale = (input_scale, input_scale)
        self.viewer.reset_view()


@click.command()
@click.option(
    "--images",
    default=REAL_DATA_ROOT / "test/images",
    help="Images path",
)
def predict(images: str):
    RegistrationAnnotator(images_root=images)
    napari.run()


if __name__ == "__main__":
    predict()
