import torch
from kornia.enhance import normalize_min_max

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class AdjustContrast:
    def __init__(self, lower_quantile: float = 0, upper_quantile: float = 0.995):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        q = torch.tensor([self.lower_quantile, self.upper_quantile])
        min_val, max_val = torch.quantile(sample.image.flatten(), q)
        sample.image = torch.clip(sample.image, min_val, max_val)
        sample.image = normalize_min_max(sample.image)
        return sample
