import pytest

from brainways.pipeline.atlas_registration import AtlasRegistration
from brainways.utils import paths
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


@pytest.fixture(autouse=True)
def mock_brainways_dir(tmp_path, monkeypatch):
    mock_dir = tmp_path / "brainways"
    mock_dir.mkdir()
    monkeypatch.setattr(paths, "_BRAINWAYS_PATH", mock_dir)


def test_trained_model_not_available(mock_atlas: BrainwaysAtlas):
    reg = AtlasRegistration(mock_atlas)
    assert not reg.is_model_available()


def test_trained_model_available(mock_rat_atlas: BrainwaysAtlas):
    reg = AtlasRegistration(mock_rat_atlas)
    assert reg.is_model_available()


# def test_atlas_registration(
#     test_data: Tuple[np.ndarray, AtlasSlice], mock_rat_atlas: BrainwaysAtlas
# ):
#     test_image, test_atlas_slice = test_data
#     reg = AtlasRegistration(mock_rat_atlas)
#     params = reg.run_automatic_registration(test_image)
#     assert np.allclose(params.ap, 321)
