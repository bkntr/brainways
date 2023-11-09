from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.transforms.depth_registration import (
    DepthRegistration,
    DepthRegistrationParams,
)
from brainways.utils._imports import BRAINWAYS_REG_MODEL_AVAILABLE
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import brain_mask, nonzero_bounding_box
from brainways.utils.paths import get_brainways_dir

if TYPE_CHECKING:
    from brainways_reg_model.model.model import BrainwaysRegModel


class AtlasRegistration:
    def __init__(self, atlas: BrainwaysAtlas):
        self.brainways_reg_model: Optional[BrainwaysRegModel] = None
        self.atlas = atlas

    def run_automatic_registration(self, image: np.ndarray):
        if not BRAINWAYS_REG_MODEL_AVAILABLE:
            raise ImportError(
                "Tried to run automatic registration but brainways_reg_model is not "
                "installed, please run `pip install brainways_reg_model` or "
                "`pip install brainways[all]`"
            )

        if not self.trained_model_available():
            raise RuntimeError(
                f"Trained model not available for {self.atlas.atlas_name}, contact "
                "Brainways team to create an automatic registration model for this atlas."
            )

        self.download_model()

        import torch
        from brainways_reg_model.model.model import BrainwaysRegModel

        if self.brainways_reg_model is None:
            self.brainways_reg_model = BrainwaysRegModel.load_from_checkpoint(
                self.local_checkpoint_path, atlas=self.atlas
            )
            self.brainways_reg_model.freeze()
        mask = brain_mask(image)
        x, y, w, h = nonzero_bounding_box(mask)
        image = image[y : y + h, x : x + w]
        params = self.brainways_reg_model.predict(torch.as_tensor(image).float())
        return params

    def automatic_registration_available(self) -> bool:
        return BRAINWAYS_REG_MODEL_AVAILABLE and self.trained_model_available()

    def get_atlas_slice(self, params: AtlasRegistrationParams) -> AtlasSlice:
        atlas_slice = self.atlas.slice(
            ap=params.ap,
            rot_horizontal=params.rot_horizontal,
            rot_sagittal=params.rot_sagittal,
            hemisphere=params.hemisphere,
        )
        return atlas_slice

    def get_transform(self, params: AtlasRegistrationParams) -> DepthRegistration:
        depth_registration_params = DepthRegistrationParams(
            td=-params.ap, rx=params.rot_sagittal, ry=params.rot_horizontal
        )
        return DepthRegistration(
            params=depth_registration_params, volume_shape=self.atlas.shape
        )

    def download_model(self):
        if self.checkpoint_downloaded():
            return
        self.local_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=f"brainways/{self.atlas.atlas_name}",
            filename="model.ckpt",
            local_dir=self.local_checkpoint_path.parent,
            local_dir_use_symlinks=False,
        )

    def checkpoint_downloaded(self) -> bool:
        return self.local_checkpoint_path.exists()

    def trained_model_available(self):
        if self.local_checkpoint_path.exists():
            return True
        api = HfApi()
        try:
            repo_files = api.list_repo_files(f"brainways/{self.atlas.atlas_name}")
            return "model.ckpt" in repo_files
        except RepositoryNotFoundError:
            return False

    @property
    def local_checkpoint_path(self) -> Path:
        return get_brainways_dir() / "reg_models" / self.atlas.atlas_name / "model.ckpt"
