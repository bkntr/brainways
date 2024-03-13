import random
from typing import Tuple

import kornia as K

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class RandomElasticDeformation:
    def __init__(self, kernel_sigma_choices: Tuple[Tuple[float, float, float, float]]):
        self.kernel_sigma_choices = kernel_sigma_choices

    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        params = random.choice(self.kernel_sigma_choices)
        kernel = tuple(params[:2])
        sigma = tuple(params[2:])

        elastic_transform = K.augmentation.RandomElasticTransform(
            kernel_size=kernel, sigma=sigma, p=1.0
        )

        params = elastic_transform.generate_parameters(sample.image.shape)
        sample.image = elastic_transform.apply_transform(
            sample.image, params, flags=elastic_transform.flags
        )
        sample.regions = elastic_transform.apply_transform(
            sample.regions,
            params,
            flags={**elastic_transform.flags, **{"mode": "nearest"}},
        )
        sample.hemispheres = elastic_transform.apply_transform(
            sample.hemispheres,
            params,
            flags={**elastic_transform.flags, **{"mode": "nearest"}},
        )
        return sample
