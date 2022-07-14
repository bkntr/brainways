import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import morphology

from brainways.utils.image import convert_to_uint8


class PreProcess:
    def __init__(self):
        self.remove_outer_speckles = RemoveOuterSpeckles()
        self.pseudo_flatfield = PseudoFlatField()

    def __call__(self, image: Image.Image):
        image_np = np.array(image).astype(np.float32)
        image_np = self.remove_outer_speckles(image_np)
        # image_np = self.pseudo_flatfield(image_np)
        output = Image.fromarray(convert_to_uint8(image_np))
        return output


class RemoveOuterSpeckles:
    def __init__(self, radius: int = 3):
        self.kernel = morphology.disk(radius)

    def __call__(self, image: np.ndarray):
        mask = (image > 10).astype(np.uint8)
        mask = morphology.opening(mask, out=mask, selem=self.kernel).astype(bool)
        image[~mask] = 0
        return image


class PseudoFlatField:
    def __init__(self, sigma: float = 5):
        self.sigma = sigma

    def __call__(self, image: np.ndarray):
        filtered_img = gaussian_filter(image, self.sigma)
        return image / (filtered_img + 1)
