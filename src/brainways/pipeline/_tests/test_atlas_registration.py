from typing import Tuple

import numpy as np
import pytest

from brainways.pipeline.atlas_registration import (
    BRAINWAYS_REG_MODEL_AVAILABLE,
    AtlasRegistration,
)
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas


def test_atlas_registration(
    test_data: Tuple[np.ndarray, AtlasSlice], mock_atlas: BrainwaysAtlas
):
    test_image, test_atlas_slice = test_data
    reg = AtlasRegistration(mock_atlas)
    if BRAINWAYS_REG_MODEL_AVAILABLE:
        reg.run_automatic_registration(test_image)
    else:
        with pytest.raises(ImportError):
            reg.run_automatic_registration(test_image)
