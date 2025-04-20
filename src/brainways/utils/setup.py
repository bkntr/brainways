import shutil
from pathlib import Path
from typing import Callable, Dict, List

import importlib_resources
import numpy as np
from PIL import Image

from brainways.pipeline.atlas_registration import AtlasRegistration
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.config import load_config, write_config
from brainways.utils.io_utils.readers.qupath_reader import QupathReader
from brainways.utils.qupath import get_brainways_qupath_dir


class BrainwaysSetup:
    def __init__(
        self, atlas_names: List[str], progress_callback: Callable[[str], None]
    ):
        self._atlas_names = atlas_names
        self._progress_callback = progress_callback
        self._downloaded_atlases: Dict[str, BrainwaysAtlas] = {}
        self._sample_image_path = (
            Path(importlib_resources.files("brainways")) / "resources/sample_image.jpg"
        )

    @staticmethod
    def is_first_launch() -> bool:
        return not load_config().initialized

    @staticmethod
    def set_initialized() -> None:
        config = load_config()
        config.initialized = True
        write_config(config)

    def run(self) -> None:
        self._progress_callback("Downloading QuPath...")
        self._download_qupath()
        for atlas_name in self._atlas_names:
            self._progress_callback(f"Downloading atlas '{atlas_name}'...")
            self._download_atlas(atlas_name)

            atlas_registration = AtlasRegistration(self._downloaded_atlases[atlas_name])
            if atlas_registration.is_model_available():
                self._progress_callback(
                    f"Downloading registration model for '{atlas_name}'..."
                )
                self._download_model(atlas_name)

        self.set_initialized()

    def _download_qupath(self) -> None:
        try:
            image = QupathReader(self._sample_image_path).get_image_data("YX")
            assert image.shape == (512, 663)
        except Exception:
            qupath_dir = get_brainways_qupath_dir()
            if qupath_dir.exists():
                shutil.rmtree(qupath_dir)
            raise

    def _download_atlas(self, atlas_name: str) -> None:
        if atlas_name in self._downloaded_atlases:
            return

        self._downloaded_atlases[atlas_name] = BrainwaysAtlas(
            atlas_name, exclude_regions=[]
        )
        _ = self._downloaded_atlases[atlas_name].reference
        _ = self._downloaded_atlases[atlas_name].annotation
        _ = self._downloaded_atlases[atlas_name].hemispheres

    def _download_model(self, atlas_name: str) -> None:
        assert atlas_name in self._downloaded_atlases
        atlas_registration = AtlasRegistration(self._downloaded_atlases[atlas_name])
        model_dir = atlas_registration.local_model_dir
        try:
            # download_model is called implicitly by run_automatic_registration if needed
            atlas_registration.run_automatic_registration(
                np.array(Image.open(self._sample_image_path))
            )
        except Exception:
            # Clean up the model directory on failure during initial run/download
            if model_dir and model_dir.exists():
                shutil.rmtree(model_dir)
            raise
