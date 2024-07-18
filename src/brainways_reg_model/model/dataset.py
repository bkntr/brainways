import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from brainglobe_atlasapi import BrainGlobeAtlas
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from brainways_reg_model.utils.config import DataConfig, load_yaml
from brainways_reg_model.utils.data import (
    get_augmentation_rotation_deg,
    model_label_to_value,
    value_to_model_label,
)

log = logging.getLogger(__name__)


class BrainwaysDataset(Dataset):
    def __init__(
        self,
        root: Path,
        data_config: DataConfig,
        augmentation: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.data_config = data_config
        self.label_params = self.data_config.label_params
        self.images_root = root / "images"
        self.labels = pd.read_csv(root / "labels.csv")
        ap_limits = self.label_params["ap"].limits
        if ap_limits is not None:
            self.labels = self.labels[
                (
                    self.labels.ap.isna()
                    | (
                        (self.labels.ap >= ap_limits[0])
                        & (self.labels.ap <= ap_limits[1])
                    )
                )
            ]
        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform
        self.metadata = load_yaml(root / "metadata.yaml")
        self.atlas = BrainGlobeAtlas(self.metadata["atlas"])
        self.images = [self.read_image(i) for i in range(len(self))]

    def read_image(self, index: int) -> torch.Tensor:
        current_raw_labels = self.labels.iloc[index]
        filename = current_raw_labels.filename
        return torch.as_tensor(
            np.array(
                Image.open(self.images_root / filename).resize(
                    self.data_config.image_size
                ),
                dtype=np.float32,
            )
        )[None, ...]

    def __getitem__(self, item):
        current_raw_labels = self.labels.iloc[item]
        current_raw_labels_mask = ~current_raw_labels.isna()

        # read label
        output_labels = {}
        output_masks = {}
        for label_name, label_params in self.label_params.items():
            value = current_raw_labels[label_name]
            value_mask = bool(current_raw_labels_mask[label_name])
            output_masks[label_name + "_mask"] = value_mask
            if value_mask:
                output_labels[label_name] = value_to_model_label(
                    value=value, label_params=label_params
                )
            else:
                output_labels[label_name] = torch.as_tensor(0)

        image = self.images[item]

        # # read structures
        # structures_path = self.images_root / f"{filename}-structures.tif"
        # if structures_path.exists():
        #     structures = Image.open(self.images_root / f"{filename}-structures.tif")
        #     structures = structure_labels(
        #         np.array(structures), self.data_config.structures, self.atlas
        #     )
        # else:
        #     structures = None
        structures = None

        # transform
        if self.transform is not None:
            image = self.transform(image[None, ...])[0]

        # target transform
        if self.target_transform is not None and structures is not None:
            structures = self.target_transform(structures)

        # augment
        if self.augmentation is not None:
            image = self.augmentation(image[None, ...])[0]
            augmentation_deg = get_augmentation_rotation_deg(self.augmentation)
            if output_masks["rot_frontal_mask"]:
                rot_frontal_value = model_label_to_value(
                    label=output_labels["rot_frontal"],
                    label_params=self.label_params["rot_frontal"],
                )
                rot_frontal_value += augmentation_deg.item()
                output_labels["rot_frontal"] = value_to_model_label(
                    value=rot_frontal_value,
                    label_params=self.label_params["rot_frontal"],
                )

        output = {"image": image, **output_labels, **output_masks}

        if structures is not None:
            output["structures"] = structures

        return output

    def __len__(self):
        return len(self.labels)


class BrainwaysDataModule(LightningDataModule):
    def __init__(
        self,
        data_paths: Dict[str, Union[str, Path]],
        data_config: DataConfig,
        num_workers: int,
        augmentation: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.data_paths = data_paths
        self.data_config = data_config
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform

    def _dataloader(self, stage: str):
        """Train/validation loaders."""
        train = stage == "train"

        data_path = self._unpack_data(self.data_paths[stage])

        dataset = BrainwaysDataset(
            root=data_path / stage,
            data_config=self.data_config,
            augmentation=self.augmentation if train else None,
            transform=self.transform if not train else None,
            target_transform=self.target_transform,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.num_workers,
            shuffle=train,
        )

    def _unpack_data(self, data_path: Union[Path, str]):
        data_path = Path(data_path)
        unpacked_data_path = data_path.with_suffix("")
        if not unpacked_data_path.exists():
            if not data_path.exists():
                raise FileNotFoundError(
                    f"{data_path} was not found. Please run `dvc pull` to download it."
                )
            shutil.unpack_archive(data_path, unpacked_data_path)
        return unpacked_data_path

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self._dataloader(stage="train")

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self._dataloader(stage="val")

    def test_dataloader(self):
        log.info("Test data loaded.")
        return self._dataloader(stage="test")
