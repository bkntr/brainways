import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy.io

from brainways.project.info_classes import MaskFileFormat


def export_mask(
    data: npt.NDArray[np.int64], path: Path, file_format: MaskFileFormat
) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    if path.suffix != f".{file_format.value}":
        path = path.with_suffix(f".{file_format.value}")

    logging.info(f"Saving {file_format.value} file to {path}")
    if file_format == MaskFileFormat.NPZ:
        np.savez_compressed(
            path,
            values=data,
        )
    elif file_format == MaskFileFormat.MAT:
        scipy.io.savemat(
            path,
            {"values": data},
            do_compression=True,
        )
    elif file_format == MaskFileFormat.CSV:
        np.savetxt(
            path,
            data,
            fmt="%d",
            delimiter=",",
        )
