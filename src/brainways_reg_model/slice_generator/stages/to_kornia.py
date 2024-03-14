import torch

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class ToKornia:
    def __call__(self, sample: SliceGeneratorSample) -> SliceGeneratorSample:
        sample.image = sample.image.to(torch.float32)[None, None]
        sample.regions = sample.regions.to(torch.float32)[None, None]
        sample.hemispheres = sample.hemispheres.to(torch.float32)[None, None]
        return sample
