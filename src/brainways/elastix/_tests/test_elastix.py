from typing import Tuple

import numpy as np

from brainways.elastix.elastix import elastix_registration
from brainways.utils.atlas.brainways_atlas import AtlasSlice


def test_elastix(test_data: Tuple[np.ndarray, AtlasSlice]):
    image, atlas_slice = test_data
    points_src = np.random.randint(0, 10, (5, 2))
    fixed = atlas_slice.reference.numpy()
    fixed_mask = None
    moving = image
    moving_mask = None
    elastix_registration(
        fixed=fixed,
        moving=moving,
        fixed_points=points_src,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
    )
