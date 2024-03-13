import random

import click
import napari
import numpy as np
import torch
from magicgui import magicgui

from brainways_reg_model.model.dataset import BrainwaysDataModule, BrainwaysDataset
from brainways_reg_model.model.model import BrainwaysRegModel
from brainways_reg_model.utils.config import load_config
from brainways_reg_model.utils.data import model_label_to_value
from brainways_reg_model.utils.paths import REAL_DATA_ZIP_PATH, SYNTH_DATA_ZIP_PATH
from brainways_reg_model.utils.slice_atlas import slice_atlas


class DatasetViewer:
    def __init__(
        self,
        dataset: BrainwaysDataset,
    ):
        self.current_idx = 0

        self.dataset = dataset
        self.viewer = napari.Viewer()
        self.atlas = self.dataset.atlas
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
            rot_horizontal={
                "label": "Horizontal Rotation",
                "widget_type": "FloatSlider",
                "min": -15,
                "max": 15,
                "enabled": False,
            },
            hemisphere={
                "label": "Hemisphere",
                "widget_type": "Slider",
                "min": 0,
                "max": 2,
                "enabled": False,
            },
        )

        self.image_slider_widget = magicgui(
            self.image_slider,
            auto_call=True,
            image_number={
                "widget_type": "Slider",
                "label": "Image #",
                "min": 1,
                "max": len(self.dataset),
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
        self.current_idx = random.randint(0, len(self.dataset) - 1)
        self.change_image()

    def registration_params(
        self,
        ap: float,
        rot_horizontal: float,
        hemisphere: str,
        valid: bool,
    ) -> napari.types.LayerDataTuple:
        pass

    def image_slider(self, image_number: int):
        image_idx = image_number - 1
        if self.current_idx != image_idx:
            self.current_idx = image_idx
            self.change_image()

    def change_image(self):
        sample = self.dataset[self.current_idx]
        image = sample["image"].numpy()
        ap = model_label_to_value(
            sample["ap"], label_params=self.dataset.label_params["ap"]
        )
        if sample["rot_frontal_mask"]:
            rot_frontal = model_label_to_value(
                sample["rot_frontal"],
                label_params=self.dataset.label_params["rot_frontal"],
            )
        else:
            rot_frontal = 0

        self.registration_params_widget(
            ap=ap,
            rot_horizontal=0,
            hemisphere=sample["hemisphere"],
            valid=bool(sample["valid"]),
            update_widget=True,
        )
        atlas_slice = slice_atlas(
            shape=self.atlas_volume.shape[1:],
            volume=self.atlas_volume,
            ap=ap,
            si=self.atlas_volume.shape[1] / 2,
            lr=self.atlas_volume.shape[2] / 2,
            rot_frontal=rot_frontal,
            rot_horizontal=0,
            rot_sagittal=0,
        ).numpy()

        self.input_layer.data = np.array(image)
        self.input_layer.reset_contrast_limits_range()
        self.input_layer.reset_contrast_limits()
        self.atlas_slice_layer.data = atlas_slice
        self.atlas_slice_layer.reset_contrast_limits_range()
        self.atlas_slice_layer.reset_contrast_limits()
        input_scale = min(
            atlas_slice.shape[0] / image.shape[0], atlas_slice.shape[1] / image.shape[1]
        )
        self.input_layer.scale = (input_scale, input_scale)
        self.viewer.reset_view()


@click.command()
@click.option(
    "--type",
    default="real",
    help="Data type - real/synth",
)
@click.option(
    "--phase",
    default="train",
    help="Data phase - train/val/test",
)
def view_dataset(type: str, phase: str):
    config = load_config("reg")

    # init model
    model = BrainwaysRegModel(config)

    zip_path = SYNTH_DATA_ZIP_PATH if type == "synth" else REAL_DATA_ZIP_PATH

    # init data
    datamodule = BrainwaysDataModule(
        data_paths={
            "train": zip_path,
            "val": zip_path,
            "test": zip_path,
        },
        data_config=config.data,
        num_workers=0,
        transform=model.transform,
        target_transform=model.target_transform,
        augmentation=model.augmentation,
    )

    dataset = None
    if phase == "train":
        dataset = datamodule.train_dataloader().dataset
    elif phase == "val":
        dataset = datamodule.val_dataloader().dataset
    elif phase == "test":
        dataset = datamodule.test_dataloader().dataset
    else:
        raise ValueError()

    DatasetViewer(dataset=dataset)
    napari.run()


if __name__ == "__main__":
    view_dataset()
