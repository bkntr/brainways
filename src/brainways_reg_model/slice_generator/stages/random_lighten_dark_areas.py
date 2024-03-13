import random
from typing import Tuple

import numpy as np
import torch

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class RandomLightenDarkAreas:
    def __init__(self, dark_value: Tuple[float, float] = (0, 0.5)):
        self.dark_value = dark_value

    def __call__(self, sample: SliceGeneratorSample):
        new_black = random.uniform(self.dark_value[0], self.dark_value[1])

        region_ids = torch.unique(
            sample.regions[(sample.regions > 0) & (sample.image < 0.2)]
        )
        max_regions_to_mask = int(np.clip(len(region_ids), 0, 20))
        num_regions_to_mask = random.randint(0, max_regions_to_mask)
        region_ids_to_mask = torch.randperm(len(region_ids))[:num_regions_to_mask]
        regions_to_mask = region_ids[region_ids_to_mask]
        mask = (sample.regions[..., None] == regions_to_mask).any(dim=-1)
        sample.image = sample.image.clone().detach()
        sample.image[mask & (sample.image < new_black)] = new_black
        return sample
