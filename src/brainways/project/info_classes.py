from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

import brainways._version
from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.utils.dataclasses import dataclass_eq
from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils import ImagePath


@dataclass(frozen=True)
class ProjectSettings:
    atlas: str
    channel: Union[int, str]


@dataclass(frozen=True)
class SubjectSettings:
    condition: Optional[str] = None
    ignore: bool = False


@dataclass(frozen=True, eq=False)
class SliceInfo:
    path: ImagePath
    image_size: ImageSizeHW
    lowres_image_size: ImageSizeHW
    params: Optional[BrainwaysParams] = BrainwaysParams()
    ignore: bool = False
    version: str = brainways._version.version

    def __eq__(self, other):
        return dataclass_eq(self, other)


class ExcelMode(Enum):
    ROW_PER_SUBJECT = auto()
    ROW_PER_IMAGE = auto()