from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.utils.dataclasses import dataclass_eq
from brainways.utils.image import ImageSizeHW
from brainways.utils.io import ImagePath


@dataclass(frozen=True)
class ProjectSettings:
    atlas: str
    channel: int


@dataclass(frozen=True, eq=False)
class ProjectDocument:
    path: ImagePath
    image_size: ImageSizeHW
    lowres_image_size: ImageSizeHW
    params: BrainwaysParams = BrainwaysParams()
    region_areas: Optional[Dict[int, int]] = None
    cells: Optional[np.ndarray] = None
    ignore: bool = False

    def __eq__(self, other):
        return dataclass_eq(self, other)
