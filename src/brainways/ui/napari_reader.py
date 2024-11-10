# example_plugin.some_module
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import napari
from napari.types import LayerData

from brainways.ui import BrainwaysUI

PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    if isinstance(path, str) and (path.endswith(".bwp") or _get_bwp_file(path)):
        return bwp_file_reader

    return None


def bwp_file_reader(path: PathOrPaths) -> List[LayerData]:
    viewer = napari.current_viewer()
    _, widget = viewer.window.add_plugin_dock_widget("brainways")
    widget: BrainwaysUI

    if path.endswith(".bwp"):
        widget.open_project_async(Path(path))
    elif _get_bwp_file(path):
        widget.open_project_async(_get_bwp_file(path))
    else:
        raise ValueError("Brainways project file not found")
    return [(None,)]


def _get_bwp_file(path: str) -> Optional[Path]:
    bwp_files = list(Path(path).glob("*.bwp"))
    if len(bwp_files) == 1:
        return bwp_files[0]
    else:
        return None
