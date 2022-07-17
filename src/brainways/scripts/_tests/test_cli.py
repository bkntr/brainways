from pathlib import Path
from unittest.mock import Mock

import numpy as np
from click.testing import CliRunner

from brainways.pipeline.cell_detector import CellDetector
from brainways.scripts.cli import cli
from brainways.utils.atlas.duracell_atlas import BrainwaysAtlas
from brainways.utils.image import ImageSizeHW
from brainways.utils.io import ImagePath


def test_cell_detection(
    mock_image_path: ImagePath, test_image_size: ImageSizeHW, tmpdir
):
    runner = CliRunner()
    CellDetector.run_cell_detector = Mock(
        return_value=np.random.randint(0, 10, test_image_size)
    )
    result = runner.invoke(
        cli,
        [
            "cell-detection",
            "--input",
            Path(mock_image_path.filename).parent,
            "--output",
            str(tmpdir),
        ],
    )
    assert result.exit_code == 0
    assert Path(tmpdir / "image_scene0.csv").exists()


def test_create_excel(project_path: Path, mock_atlas: BrainwaysAtlas, tmpdir):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "create-excel",
        ],
    )
    assert result.exit_code == 2
