from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from brainways.pipeline.atlas_registration import AtlasRegistration
from brainways.utils import paths
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas


@pytest.fixture(autouse=True)
def mock_brainways_dir(tmpdir, monkeypatch):
    monkeypatch.setattr(paths, "_BRAINWAYS_PATH", Path(tmpdir))


@pytest.fixture
def mock_rat_atlas(mock_atlas) -> BrainwaysAtlas:
    mock_atlas.brainglobe_atlas.atlas_name = "whs_sd_rat_39um"
    return mock_atlas


def test_trained_model_not_available(mock_atlas: BrainwaysAtlas):
    reg = AtlasRegistration(mock_atlas)
    assert not reg.trained_model_available()


def test_trained_model_available(mock_rat_atlas: BrainwaysAtlas):
    reg = AtlasRegistration(mock_rat_atlas)
    assert reg.trained_model_available()


def test_download_model(mock_rat_atlas: BrainwaysAtlas, tmpdir):
    reg = AtlasRegistration(mock_rat_atlas)
    assert not reg.checkpoint_downloaded()
    reg.download_model()
    assert reg.checkpoint_downloaded()


def test_atlas_registration(
    test_data: Tuple[np.ndarray, AtlasSlice], mock_rat_atlas: BrainwaysAtlas
):
    test_image, test_atlas_slice = test_data
    reg = AtlasRegistration(mock_rat_atlas)
    params = reg.run_automatic_registration(test_image)
    assert np.allclose(params.ap, 319.017578125)
