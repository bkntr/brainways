from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import brainways._version
from brainways.pipeline.brainways_params import BrainwaysParams, CellDetectorParams
from brainways.utils.dataclasses import dataclass_eq
from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers import QupathReader


@dataclass(frozen=True)
class ProjectSettings:
    atlas: str
    channel: Union[int, str]
    default_cell_detector_params: CellDetectorParams = CellDetectorParams()
    condition_names: List[str] = field(default_factory=list)
    version: str = brainways._version.version


@dataclass(frozen=True)
class SubjectInfo:
    name: str
    conditions: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class SliceInfo:
    path: ImagePath
    image_size: ImageSizeHW
    lowres_image_size: ImageSizeHW
    params: BrainwaysParams = BrainwaysParams()
    ignore: bool = False
    physical_pixel_sizes: Optional[Tuple[float, float]] = None

    def image_reader(self) -> QupathReader:
        reader = QupathReader(self.path.filename)
        reader.set_scene(self.path.scene)
        return reader

    def __eq__(self, other):
        return dataclass_eq(self, other)


@dataclass(frozen=True)
class SubjectFileFormat:
    subject_info: SubjectInfo
    slice_infos: List[SliceInfo]


class ExcelMode(Enum):
    ROW_PER_SUBJECT = auto()
    ROW_PER_IMAGE = auto()
