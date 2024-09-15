from __future__ import annotations

import numpy as np

from brainways.pipeline.brainways_params import AffineTransform2DParams
from brainways.transforms.affine_transform_2d import BrainwaysAffineTransform2D
from brainways.utils.atlas.brainways_atlas import AtlasSlice
from brainways.utils.image import ImageSizeHW, brain_mask, nonzero_bounding_box


class Affine2D:
    def find_transformation_params(
        self, image: np.ndarray, atlas_slice: AtlasSlice
    ) -> AffineTransform2DParams:
        box1 = nonzero_bounding_box(brain_mask(image))
        box2 = nonzero_bounding_box(brain_mask(atlas_slice.reference.numpy()))

        return BrainwaysAffineTransform2D.find_transform_between_boxes(
            box1, box2, input_size=image.shape
        ).params

    def get_transform(
        self,
        params: AffineTransform2DParams,
        input_size: ImageSizeHW,
        scale: float | None,
    ) -> BrainwaysAffineTransform2D:
        return BrainwaysAffineTransform2D(
            params=params, input_size=input_size, scale=scale
        )
