import json
import shutil
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from brainways.model.model_utils import load_model
from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.transforms.depth_registration import (
    DepthRegistration,
    DepthRegistrationParams,
)
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.paths import get_brainways_dir

if TYPE_CHECKING:
    from brainways.model.siamese.siamese_model import SiameseModel

_MODEL_FILES: Tuple[str, ...] = ("config.yml", "state_dict.pt")
_SIAMESE_REPO_ID = "brainways/siamese"
_MODEL_IDS_FILENAME = "model_ids.json"


class AtlasRegistration:
    def __init__(self, atlas: BrainwaysAtlas):
        self.model: Optional[SiameseModel] = None
        self.atlas = atlas

    def run_automatic_registration(self, image: np.ndarray) -> AtlasRegistrationParams:
        if not self.is_model_available():
            raise RuntimeError(
                f"Trained model not available for {self.atlas.atlas_name}, contact Brainways team"
            )

        if self.model is None:
            self.download_model()
            self.model = load_model(self.local_model_dir)

        pred_ap, _ = self.model.predict(image, atlas_name=self.atlas.atlas_name)
        ap_val = float(pred_ap)
        return AtlasRegistrationParams(ap=ap_val)

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
        if self._model_id is None:
            raise ValueError(
                f"Failed to download model files for atlas {self.atlas.atlas_name}: No model ID found. "
                "Is there a model available for this atlas?"
            )

        model_dir = self.local_model_dir

        if self.is_model_downloaded():
            return

        model_dir.mkdir(parents=True, exist_ok=True)

        # Download config and state from the specific model ID subfolder
        try:
            for filename in _MODEL_FILES:
                hf_hub_download(
                    repo_id=_SIAMESE_REPO_ID,
                    filename=filename,
                    subfolder=self._model_id,
                    local_dir=self._reg_models_dir(),
                    local_dir_use_symlinks=False,
                )
        except Exception as e:
            # Clean up potentially partially downloaded files/directory on error
            if model_dir.exists():
                shutil.rmtree(model_dir)
            # Re-raise the exception after cleanup attempt
            raise RuntimeError(
                f"Failed to download model files for atlas {self.atlas.atlas_name} (model ID: {self._model_id}): {e}"
            ) from e

    def is_model_downloaded(self) -> bool:
        if self._model_id is None:
            return False

        model_dir = self.local_model_dir
        if not model_dir.exists():
            return False
        return all((model_dir / filename).exists() for filename in _MODEL_FILES)

    def is_model_available(self) -> bool:
        if self._model_id is None:
            return False

        if self.is_model_downloaded():
            return True

        # Check if all required files are present in the remote repo subfolder
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=_SIAMESE_REPO_ID, revision="main")
        return all(f"{self._model_id}/{fname}" in repo_files for fname in _MODEL_FILES)

    @property
    def local_model_dir(self) -> Path:
        if self._model_id is None:
            raise ValueError(
                "Model ID is not set. Cannot determine local model directory."
            )
        return AtlasRegistration._reg_models_dir() / self._model_id

    @cached_property
    def _model_id(self) -> Optional[str]:
        """Loads the model ID mapping from the local cache or downloads it."""
        models_dir = AtlasRegistration._reg_models_dir()
        local_ids_path = models_dir / _MODEL_IDS_FILENAME
        if not local_ids_path.exists():
            try:
                hf_hub_download(
                    repo_id=_SIAMESE_REPO_ID,
                    filename=_MODEL_IDS_FILENAME,
                    local_dir=models_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {_MODEL_IDS_FILENAME}: {e}"
                ) from e

        try:
            with open(local_ids_path) as f:
                result = json.load(f)
            # Ensure the loaded result is a dictionary
            if not isinstance(result, dict):
                raise ValueError(
                    f"Invalid format in {local_ids_path}: expected a dictionary."
                )

            return result.get(self.atlas.atlas_name)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # If loading fails (e.g., corrupted file), attempt to remove and raise
            if local_ids_path.exists():
                local_ids_path.unlink()
            raise RuntimeError(f"Failed to load or parse {local_ids_path}: {e}") from e

    @staticmethod
    def _reg_models_dir() -> Path:
        return get_brainways_dir() / "reg_models"
