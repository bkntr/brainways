import random
from typing import Tuple

import kornia as K
import torch
from kornia.filters import get_gaussian_kernel2d

from brainways_reg_model.slice_generator.slice_generator_sample import (
    SliceGeneratorSample,
)


class RandomLightDeformation:
    def __init__(
        self, kernel_sigma_choices: Tuple[Tuple[float, float, float, float], ...]
    ):
        self.kernel_sigma_choices = kernel_sigma_choices

    def __call__(self, sample: SliceGeneratorSample):
        params = random.choice(self.kernel_sigma_choices)
        kernel_size = tuple(params[:2])
        sigma = tuple(params[2:])
        alpha = 6.0

        B, _, H, W = sample.image.shape
        noise = torch.rand(B, 2, H, W) * 2 - 1

        # Get Gaussian kernel for 'y' and 'x' displacement
        kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, (sigma[0], sigma[0]))[
            None
        ]

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp: torch.Tensor = noise[:, :1]
        disp = K.filters.filter2d(disp, kernel=kernel, border_type="constant") * alpha

        sample.image = torch.clip(sample.image + disp, min=0, max=1)
        return sample
