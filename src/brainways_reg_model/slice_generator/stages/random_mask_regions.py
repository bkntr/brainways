import random

import numpy as np
import torch

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class RandomMaskRegions:
    def __init__(self):
        pass

    def __call__(self, sample: SliceGeneratorSample):
        region_ids = torch.unique(sample.regions)
        max_regions_to_mask = int(np.clip(len(region_ids) - 3, 0, 20))
        num_regions_to_mask = random.randint(0, max_regions_to_mask)
        region_ids_to_mask = torch.randperm(len(region_ids))[:num_regions_to_mask]
        regions_to_mask = region_ids[region_ids_to_mask]
        sample.image = sample.image * (
            sample.regions[..., None] != regions_to_mask
        ).all(dim=-1)

        return sample
