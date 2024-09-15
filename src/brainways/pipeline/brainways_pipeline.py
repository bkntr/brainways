from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto

import cv2
import numpy as np
import skimage.transform

from brainways.pipeline.affine_2d import Affine2D
from brainways.pipeline.atlas_registration import AtlasRegistration
from brainways.pipeline.brainways_params import AffineTransform2DParams, BrainwaysParams
from brainways.pipeline.tps import TPS
from brainways.project.info_classes import SliceInfo
from brainways.transforms.base import BrainwaysTransform
from brainways.transforms.compose import Compose
from brainways.transforms.identity_transform import IdentityTransform
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import ImageSizeHW, convert_to_uint8


class PipelineStep(Enum):
    AFFINE_2D = auto()
    TPS = auto()
    ATLAS_REGISTRATION = auto()


class BrainwaysPipeline:
    def __init__(self, atlas: BrainwaysAtlas):
        self.atlas = atlas
        self.atlas_registration = AtlasRegistration(atlas=atlas)
        self.affine_2d = Affine2D()
        self.tps = TPS()

    def get_atlas_slice(self, brainways_params: BrainwaysParams) -> AtlasSlice:
        return self.atlas_registration.get_atlas_slice(brainways_params.atlas)

    def get_image_to_atlas_transform(
        self,
        brainways_params: BrainwaysParams,
        lowres_image_size: ImageSizeHW,
        until_step: PipelineStep | None = None,
        scale: float | None = None,
    ) -> BrainwaysTransform:
        until_step_value = 1000 if until_step is None else until_step.value
        affine_2d_transform: BrainwaysTransform = IdentityTransform()
        tps_transform: BrainwaysTransform = IdentityTransform()
        atlas_registration_transform: BrainwaysTransform = IdentityTransform()
        if (
            until_step_value >= PipelineStep.AFFINE_2D.value
            and brainways_params.affine is not None
        ):
            affine_2d_transform = self.affine_2d.get_transform(
                brainways_params.affine,
                input_size=lowres_image_size,
                scale=scale,
            )
        if (
            until_step_value >= PipelineStep.TPS.value
            and brainways_params.tps is not None
        ):
            tps_transform = self.tps.get_transform(brainways_params.tps, scale=scale)
        if (
            until_step_value >= PipelineStep.ATLAS_REGISTRATION.value
            and brainways_params.atlas is not None
        ):
            atlas_registration_transform = self.atlas_registration.get_transform(
                brainways_params.atlas
            )

        image_to_atlas_transform = Compose(
            [
                affine_2d_transform,
                tps_transform,
                atlas_registration_transform,
            ]
        )
        return image_to_atlas_transform

    def find_2d_affine_transform(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> AffineTransform2DParams:
        atlas_slice = self.get_atlas_slice(params)
        image = skimage.transform.rotate(
            image, angle=-params.atlas.rot_frontal, mode="reflect"
        )
        affine_params = self.affine_2d.find_transformation_params(
            image=image, atlas_slice=atlas_slice
        )
        return replace(affine_params, angle=params.atlas.rot_frontal)

    def transform_image(
        self,
        image: np.ndarray,
        params: BrainwaysParams,
        until_step: PipelineStep | None = None,
        scale: float | None = None,
    ):
        scale = scale or 1.0
        transform = self.get_image_to_atlas_transform(
            brainways_params=params,
            lowres_image_size=image.shape,
            until_step=until_step,
            scale=scale,
        )

        output_size = (
            int(self.atlas.shape[1] * scale),
            int(self.atlas.shape[2] * scale),
        )

        transformed_image = transform.transform_image(
            image=image, output_size=output_size
        )

        if image.dtype == np.uint8:
            transformed_image = convert_to_uint8(transformed_image)

        return transformed_image.astype(image.dtype)

    def get_registered_annotation_on_image(self, slice_info: SliceInfo):
        annotation = np.array(
            self.get_atlas_slice(slice_info.params).annotation
        ).astype(np.float32)
        transform = self.get_image_to_atlas_transform(
            brainways_params=slice_info.params,
            lowres_image_size=slice_info.lowres_image_size,
        ).inv()
        # transformed_annotation = transform.transform_image(annotation, output_size=slice_info.image_size, mode="nearest")
        transformed_annotation = transform.transform_image(
            annotation, output_size=slice_info.lowres_image_size, mode="nearest"
        )
        transformed_annotation = cv2.resize(
            transformed_annotation,
            (slice_info.image_size[1], slice_info.image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)
        return transformed_annotation
