import random
from typing import Tuple

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class RandomZeroBelowThreshold:
    def __init__(self, threshold: Tuple[float, float]):
        self.threshold = threshold

    def __call__(self, sample: SliceGeneratorSample):
        threshold = random.uniform(self.threshold[0], self.threshold[1])
        sample.image = sample.image * (sample.image >= threshold)
        return sample
