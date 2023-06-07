from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from brainways.utils.dataclasses import dataclass_eq


@dataclass(frozen=True)
class BrainwaysParams:
    atlas: Optional[AtlasRegistrationParams] = None
    affine: Optional[AffineTransform2DParams] = None
    tps: Optional[TPSTransformParams] = None
    cell: Optional[CellDetectorParams] = None


@dataclass
class CellDetectorParams:
    normalizer: str = "quantile"
    normalizer_range: Tuple[float, float] = (0.98, 0.997)


@dataclass
class AffineTransform2DParams:
    angle: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    cx: Optional[float] = None
    cy: Optional[float] = None


@dataclass(frozen=True)
class TPSTransformParams:
    points_src: np.ndarray
    points_dst: np.ndarray

    def __eq__(self, other):
        return dataclass_eq(self, other)


@dataclass(frozen=True)
class AtlasRegistrationParams:
    ap: float = 0.0
    rot_frontal: float = 0.0
    rot_horizontal: float = 0.0
    rot_sagittal: float = 0.0
    hemisphere: str = "both"
    confidence: float = 1.0
