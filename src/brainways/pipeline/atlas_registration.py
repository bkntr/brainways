from typing import TYPE_CHECKING, Optional

import numpy as np

from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.transforms.depth_registration import (
    DepthRegistration,
    DepthRegistrationParams,
)
from brainways.utils._imports import BRAINWAYS_REG_MODEL_AVAILABLE
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import brain_mask, nonzero_bounding_box
from brainways.utils.paths import REG_MODEL

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

        import torch
        from brainways_reg_model.model.model import BrainwaysRegModel

        if self.brainways_reg_model is None:
            self.brainways_reg_model = BrainwaysRegModel.load_from_checkpoint(
                REG_MODEL, atlas=self.atlas
            )
            self.brainways_reg_model.freeze()
        mask = brain_mask(image)
        x, y, w, h = nonzero_bounding_box(mask)
        image = image[y : y + h, x : x + w]
        params = self.brainways_reg_model.predict(torch.as_tensor(image).float())
        return params

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
