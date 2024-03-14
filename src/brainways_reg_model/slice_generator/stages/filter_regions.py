from typing import List, Optional

import torch

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class FilterRegions:
    def __init__(self, exclude: Optional[List[int]] = None):
        self.exclude = torch.tensor(exclude) if exclude else None

    def __call__(self, sample: SliceGeneratorSample):
        exclude_mask = (sample.regions[..., None] != self.exclude).all(dim=-1)
        sample.image = sample.image * exclude_mask
        # sample.regions = sample.regions * exclude_mask

        return sample
