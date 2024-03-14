from kornia.utils import ImageToTensor

from brainways_reg_model.slice_generator.stages.adjust_contrast import AdjustContrast
from brainways_reg_model.slice_generator.stages.crop_material_area import (
    CropMaterialArea,
)
from brainways_reg_model.slice_generator.stages.filter_regions import FilterRegions
from brainways_reg_model.slice_generator.stages.random_affine import RandomAffine
from brainways_reg_model.slice_generator.stages.random_elastic_deformation import (
    RandomElasticDeformation,
)
from brainways_reg_model.slice_generator.stages.random_light_deformation import (
    RandomLightDeformation,
)
from brainways_reg_model.slice_generator.stages.random_lighten_dark_areas import (
    RandomLightenDarkAreas,
)
from brainways_reg_model.slice_generator.stages.random_mask_regions import (
    RandomMaskRegions,
)
from brainways_reg_model.slice_generator.stages.random_single_hemisphere import (
    RandomSingleHemisphere,
)
from brainways_reg_model.slice_generator.stages.random_zero_below_threshold import (
    RandomZeroBelowThreshold,
)
from brainways_reg_model.slice_generator.stages.resize import Resize
from brainways_reg_model.slice_generator.stages.to_kornia import ToKornia
from brainways_reg_model.slice_generator.stages.to_pil_image import ToPILImage

stages_dict = {
    "crop_material_area": CropMaterialArea,
    "to_tensor": ImageToTensor,
    "random_affine": RandomAffine,
    "random_elastic_deformation": RandomElasticDeformation,
    "random_zero_below_threshold": RandomZeroBelowThreshold,
    "random_mask_regions": RandomMaskRegions,
    "random_lighten_dark_areas": RandomLightenDarkAreas,
    "random_light_deformation": RandomLightDeformation,
    "adjust_contrast": AdjustContrast,
    "resize": Resize,
    "to_kornia": ToKornia,
    "to_pil_image": ToPILImage,
    "random_single_hemisphere": RandomSingleHemisphere,
    "filter_regions": FilterRegions,
}
