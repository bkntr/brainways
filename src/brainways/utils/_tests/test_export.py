from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import scipy.io

from brainways.project.info_classes import MaskFileFormat
from brainways.utils.export import export_mask


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1, 2], [3, 4]]),
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    ],
)
@pytest.mark.parametrize(
    "file_format, extension, loader",
    [
        pytest.param(
            MaskFileFormat.NPZ,
            "npz",
            lambda x: np.load(x)["values"],
            id="npz",
        ),
        pytest.param(
            MaskFileFormat.CSV,
            "csv",
            lambda x: np.loadtxt(x, delimiter=","),
            id="csv",
        ),
        pytest.param(
            MaskFileFormat.MAT,
            "mat",
            lambda x: scipy.io.loadmat(x)["values"],
            id="mat",
        ),
    ],
)
def test_export_mask(
    values: np.ndarray,
    file_format: MaskFileFormat,
    extension: str,
    loader: Callable[[Path], np.ndarray],
    tmp_path: Path,
):
    output_dir = tmp_path / "output"

    if file_format == MaskFileFormat.CSV and values.ndim == 3:
        with pytest.raises(ValueError):
            export_mask(data=values, path=output_dir / "test", file_format=file_format)
        return
    else:
        export_mask(data=values, path=output_dir / "test", file_format=file_format)

    output_file = output_dir / f"test.{extension}"
    assert output_file.exists()
    data = loader(output_file)
    assert np.array_equal(data, values)
