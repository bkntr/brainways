from typing import Tuple

import numpy as np

from brainways.pipeline.atlas_registration import AtlasRegistration
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas


def test_atlas_registration(
    test_data: Tuple[np.ndarray, AtlasSlice], mock_atlas: BrainwaysAtlas
):
    test_image, test_atlas_slice = test_data
    reg = AtlasRegistration(mock_atlas)
    reg.run_automatic_registration(test_image)
