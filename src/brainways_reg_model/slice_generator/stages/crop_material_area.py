from typing import Tuple

import kornia as K
import torch
from kornia.geometry.transform.crop2d import crop_and_resize

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class CropMaterialArea:
    def __init__(
        self,
        ignore_values_under: float = 3 / 255,
        open_kernel_size: Tuple[int, int] = (5, 5),
        pad: int = 2,
    ):
        self.ignore_values_under = ignore_values_under
        self.open_kernel_size = open_kernel_size
        self.pad = pad

    @staticmethod
    def nonzero_bounding_box(image: torch.Tensor):
        _, _, ys, xs = torch.nonzero(image).T
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        return x0, y0, x1 - x0, y1 - y0

    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        image_ignored_small_values = sample.image * (
            sample.image > self.ignore_values_under
        )

        # if image is all zeros, do nothing
        if (image_ignored_small_values == 0).all():
            return sample

        x, y, w, h = self.nonzero_bounding_box(image_ignored_small_values)
        boxes = K.geometry.bbox_generator(
            x - self.pad, y - self.pad, w + self.pad * 2, h + self.pad * 2
        )
        sample.image = crop_and_resize(
            sample.image,
            boxes,
            sample.image.shape[2:],
            align_corners=True,
        )
        sample.regions = crop_and_resize(
            sample.regions,
            boxes,
            sample.image.shape[2:],
            mode="nearest",
            align_corners=True,
        )
        sample.hemispheres = crop_and_resize(
            sample.hemispheres,
            boxes,
            sample.image.shape[2:],
            mode="nearest",
            align_corners=True,
        )
        return sample
