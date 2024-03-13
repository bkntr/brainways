try:
    from napari_brainways._version import version as __version__
except ImportError:
    __version__ = "unknown"
from napari_brainways._sample_project import (
    load_sample_project,
    load_sample_project_annotated,
)
from napari_brainways.brainways_ui import BrainwaysUI

__all__ = (
    "load_sample_project",
    "load_sample_project_annotated",
    "BrainwaysUI",
)
