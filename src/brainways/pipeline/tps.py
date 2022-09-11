from __future__ import annotations

from dataclasses import replace

import numpy as np

from brainways.elastix.elastix import elastix_registration
from brainways.pipeline.brainways_params import BrainwaysParams, TPSTransformParams
from brainways.transforms.tps_transform import TPSTransform
from brainways.utils.atlas.brainways_atlas import AtlasSlice


class TPS:
    def find_registration_params(
        self,
        image: np.ndarray,
        atlas_slice: AtlasSlice,
        params: BrainwaysParams,
    ) -> BrainwaysParams:
        # TODO: comments
        fixed = atlas_slice.reference.numpy()
        fixed_mask = None
        moving = image
        moving_mask = None
        registered_src_points = elastix_registration(
            fixed=fixed,
            moving=moving,
            fixed_points=params.tps.points_src,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
        )
        transform = TPSTransform(
            TPSTransformParams(
                points_src=registered_src_points,
                points_dst=params.tps.points_src,
            )
        )
        registered_dst_points = transform.transform_points(params.tps.points_src)
        return replace(
            params,
            tps=TPSTransformParams(
                points_src=params.tps.points_src,
                points_dst=registered_dst_points,
            ),
        )

    def get_transform(self, params: TPSTransformParams) -> TPSTransform:
        return TPSTransform(params=params)
