from dataclasses import dataclass
from pathlib import Path
from typing import Union

import dask.array
import numpy as np


@dataclass
class ImageEntry:
    image: Union[np.ndarray, dask.array.Array]
    path: Path
    scene: int
