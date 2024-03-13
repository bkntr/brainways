from typing import Tuple

from kornia.geometry.transform import resize

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)
from brainways_reg_model.utils.config import load_config


class Resize:
    def __init__(self, size: Tuple[int, int] = None):
        if size is None:
            size = load_config().data.image_size
        self.size = size

    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        sample.image = resize(sample.image, self.size)
        sample.regions = resize(sample.regions, self.size, interpolation="nearest")
        sample.hemispheres = resize(
            sample.hemispheres, self.size, interpolation="nearest"
        )
        return sample
