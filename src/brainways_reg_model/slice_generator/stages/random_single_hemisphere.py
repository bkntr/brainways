import numpy as np

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class RandomSingleHemisphere:
    def generate_parameters(self):
        return np.random.choice((0, 1, 2), p=(0.75, 0.125, 0.125))

    def __call__(self, sample: SliceGeneratorSample):
        mode = self.generate_parameters()

        if mode == 1:
            sample.image *= sample.hemispheres == 1
            sample.regions *= sample.hemispheres == 1
        elif mode == 2:
            sample.image *= sample.hemispheres == 2
            sample.regions *= sample.hemispheres == 2

        return sample
