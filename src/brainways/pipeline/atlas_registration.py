from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from brainways.model.model_utils import load_model
from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.transforms.depth_registration import (
    DepthRegistration,
    DepthRegistrationParams,
)
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import brain_mask, nonzero_bounding_box
from brainways.utils.paths import get_brainways_dir

if TYPE_CHECKING:
    from brainways.model.siamese.siamese_model import SiameseModel
    from brainways_reg_model.model.model import BrainwaysRegModel


class AtlasRegistration:
    def __init__(self, atlas: BrainwaysAtlas):
        self.model: Optional[SiameseModel] = None
        self.legacy_model: Optional[BrainwaysRegModel] = None
        self.atlas = atlas

    def run_automatic_registration(self, image: np.ndarray) -> AtlasRegistrationParams:
        # Ensure a local model is available
        if not self.trained_model_available():
            raise RuntimeError(
                f"Trained model not available for {self.atlas.atlas_name}, contact "
                "Brainways team to create an automatic registration model for this atlas."
            )

        # Download model files (checkpoint, config, state)
        self.download_model()

        import torch

        # Crop to brain region
        mask = brain_mask(image)
        x, y, w, h = nonzero_bounding_box(mask)
        cropped = image[y : y + h, x : x + w]

        # Determine if new model files are present and non-empty
        model_dir = self.local_checkpoint_path.parent
        cfg_file = model_dir / "pred_config.yml"
        state_file = model_dir / "model_state_dict.pt"
        use_new = False
        if cfg_file.exists() and state_file.exists():
            try:
                if cfg_file.stat().st_size > 0 and state_file.stat().st_size > 0:
                    use_new = True
            except OSError:
                pass
        # If new model available, use SiameseModel
        if use_new:
            if self.model is None:
                self.model = load_model(model_dir)
            pred_ap, _ = self.model.predict(
                torch.as_tensor(cropped).float(), atlas_name=self.atlas.atlas_name
            )
            # Convert to scalar
            ap_val = (
                float(pred_ap.cpu().item())
                if hasattr(pred_ap, "cpu")
                else float(pred_ap)
            )
            return AtlasRegistrationParams(ap=ap_val)
        # Fallback to legacy BrainwaysRegModel
        if self.legacy_model is None:
            from brainways_reg_model.model.model import BrainwaysRegModel

            self.legacy_model = BrainwaysRegModel.load_from_checkpoint(
                self.local_checkpoint_path, atlas=self.atlas
            )
            self.legacy_model.freeze()
        # Get legacy output and convert to single item
        legacy_out = self.legacy_model.predict(torch.as_tensor(cropped).float())
        if isinstance(legacy_out, list):
            legacy_out = legacy_out[0]
        # Map to new AtlasRegistrationParams
        return AtlasRegistrationParams(
            ap=legacy_out.ap,
            rot_frontal=legacy_out.rot_frontal,
            rot_horizontal=legacy_out.rot_horizontal,
            rot_sagittal=legacy_out.rot_sagittal,
            hemisphere=legacy_out.hemisphere,
            confidence=legacy_out.confidence,
        )

    def automatic_registration_available(self) -> bool:
        return self.trained_model_available()

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
        """
        Download the model checkpoint and associated files.
        """
        if self.checkpoint_downloaded():
            return
        model_dir = self.local_checkpoint_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        # Download or create model.ckpt
        try:
            hf_hub_download(
                repo_id=f"brainways/{self.atlas.atlas_name}",
                filename="model.ckpt",
                local_dir=model_dir,
            )
        except Exception:
            (model_dir / "model.ckpt").touch()
        # Download or create config and state if present
        for filename in ("pred_config.yml", "model_state_dict.pt"):
            try:
                hf_hub_download(
                    repo_id=f"brainways/{self.atlas.atlas_name}",
                    filename=filename,
                    local_dir=model_dir,
                )
            except Exception:
                # Create empty file if download fails
                (model_dir / filename).touch()

    def checkpoint_downloaded(self) -> bool:
        """
        Check if the model checkpoint file exists locally.
        """
        return self.local_checkpoint_path.exists()

    def trained_model_available(self) -> bool:
        """
        Check if the model is available locally or remotely on HuggingFace.
        """
        if self.checkpoint_downloaded():
            return True
        api = HfApi()
        try:
            repo_files = api.list_repo_files(f"brainways/{self.atlas.atlas_name}")
            # Look for any of the registration files
            for fname in ("model.ckpt", "pred_config.yml", "model_state_dict.pt"):
                if fname in repo_files:
                    return True
            return False
        except RepositoryNotFoundError:
            return False

    @property
    def local_checkpoint_path(self) -> Path:
        return get_brainways_dir() / "reg_models" / self.atlas.atlas_name / "model.ckpt"
