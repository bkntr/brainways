try:
    from brainways.ui._version import version as __version__
except ImportError:
    __version__ = "unknown"
from brainways.ui._sample_project import (
    load_sample_project,
    load_sample_project_annotated,
)
from brainways.ui.brainways_ui import BrainwaysUI

__all__ = (
    "load_sample_project",
    "load_sample_project_annotated",
    "BrainwaysUI",
)
