from distutils.dir_util import copy_tree
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from brainways.pipeline.atlas_registration import AtlasRegistration
from brainways.utils import paths
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.io_utils.readers.qupath_reader import QupathReader
from brainways.utils.setup import BrainwaysSetup

_TEST_ATLAS = "whs_sd_rat_39um"


@pytest.fixture
def setup(mock_rat_atlas: BrainwaysAtlas) -> BrainwaysSetup:
    setup = BrainwaysSetup(atlas_names=[_TEST_ATLAS], progress_callback=Mock())
    setup._downloaded_atlases = {_TEST_ATLAS: mock_rat_atlas}
    setup.run()
    return setup


def test_setup_outputs_progress(setup: BrainwaysSetup):
    setup._progress_callback.assert_called()  # type: ignore


def test_download_model_success(setup: BrainwaysSetup, mock_rat_atlas: BrainwaysAtlas):
    atlas_registration = AtlasRegistration(mock_rat_atlas)
    assert atlas_registration.checkpoint_downloaded()


def test_download_model_fails(
    setup: BrainwaysSetup,
    mock_rat_atlas: BrainwaysAtlas,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    copy_tree(str(paths._BRAINWAYS_PATH / "reg_models"), str(tmp_path / "reg_models"))
    monkeypatch.setattr(paths, "_BRAINWAYS_PATH", tmp_path)
    atlas_registration = AtlasRegistration(mock_rat_atlas)

    with open(atlas_registration.local_checkpoint_path, "wb") as f:
        f.seek(0)
        f.write(np.zeros(100).tobytes())

    assert atlas_registration.checkpoint_downloaded()

    with pytest.raises(Exception):
        setup._download_model(_TEST_ATLAS)

    assert not atlas_registration.checkpoint_downloaded()


def test_download_qupath_fails(
    setup: BrainwaysSetup,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    tmp_qupath_path = tmp_path / "qupath"

    copy_tree(str(paths._BRAINWAYS_PATH / "qupath"), str(tmp_qupath_path))
    monkeypatch.setattr(paths, "_BRAINWAYS_PATH", tmp_path)

    assert tmp_qupath_path.exists()

    with pytest.raises(Exception):

        def raise_error():
            raise AssertionError

        monkeypatch.setattr(QupathReader, "_initialize_qupath", raise_error)
        setup._download_qupath()

    assert not tmp_qupath_path.exists()


def test_is_first_launch(
    setup: BrainwaysSetup,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(paths, "_BRAINWAYS_PATH", tmp_path)
    copy_tree(str(paths._BRAINWAYS_PATH), str(tmp_path))
    paths.get_brainways_config_path().unlink(missing_ok=True)
    assert setup.is_first_launch()
    setup.run()
    assert not setup.is_first_launch()
