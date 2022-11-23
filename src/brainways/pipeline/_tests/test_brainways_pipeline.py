from brainways.pipeline.brainways_params import AtlasRegistrationParams, BrainwaysParams
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


def test_find_2d_affine_transform(test_data, mock_atlas: BrainwaysAtlas):
    image, _ = test_data
    params = BrainwaysParams(atlas=AtlasRegistrationParams(rot_frontal=10.0))
    pipeline = BrainwaysPipeline(mock_atlas)
    affine_params = pipeline.find_2d_affine_transform(image, params)
    assert affine_params.angle == 10
