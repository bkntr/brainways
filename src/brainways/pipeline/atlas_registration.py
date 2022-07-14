from typing import TYPE_CHECKING, Optional

import numpy as np
import PIL.Image

from brainways.pipeline.brainways_params import AtlasRegistrationParams
from brainways.transforms.depth_registration import (
    DepthRegistration,
    DepthRegistrationParams,
)
from brainways.utils._imports import BRAINWAYS_REG_MODEL_AVAILABLE
from brainways.utils.atlas.duracell_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import convert_to_uint8
from brainways.utils.paths import REG_MODEL

if TYPE_CHECKING:
    from brainways_reg_model import BrainwaysRegModel


class AtlasRegistration:
    def __init__(self, atlas: BrainwaysAtlas):
        self.duracell_reg_model: Optional[BrainwaysRegModel] = None
        self.atlas = atlas

    def run_automatic_registration(self, image: np.ndarray):
        if not BRAINWAYS_REG_MODEL_AVAILABLE:
            raise ImportError(
                "Tried to run automatic registration but brainways_reg_model is not "
                "installed, please run `pip install brainways_reg_model` or "
                "`pip install brainways[all]`"
            )

        from brainways_reg_model import BrainwaysRegModel

        if self.duracell_reg_model is None:
            self.duracell_reg_model = BrainwaysRegModel.load_from_checkpoint(
                REG_MODEL, atlas=self.atlas
            )
            self.duracell_reg_model.freeze()
        image = np.clip(image, image.min(), np.quantile(image, 0.998))
        image = convert_to_uint8(image)
        params = self.duracell_reg_model.predict(PIL.Image.fromarray(image))
        return params

    def get_atlas_slice(self, params: AtlasRegistrationParams) -> AtlasSlice:
        atlas_slice = self.atlas.slice(
            ap=params.ap,
            rot_horizontal=params.rot_horizontal,
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
