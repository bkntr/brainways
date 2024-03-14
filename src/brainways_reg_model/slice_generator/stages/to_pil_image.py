import kornia as K
import numpy as np
from PIL import Image

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class ToPILImage:
    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        sample.image = K.utils.tensor_to_image(sample.image)
        sample.image = Image.fromarray((sample.image * 255).astype(np.uint8), mode="L")
        sample.regions = K.utils.tensor_to_image(sample.regions.int())
        sample.regions = Image.fromarray(sample.regions, mode="I")
        return sample
