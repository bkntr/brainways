from pathlib import Path

from click.testing import CliRunner

from brainways.scripts.cli import cli
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas

# def test_cell_detection(
#     project_path: Path, test_image_size: ImageSizeHW, tmpdir
# ):
#     runner = CliRunner()
#     CellDetector.run_cell_detector = Mock(
#         return_value=np.random.randint(0, 10, test_image_size)
#     )
#     result = runner.invoke(
#         cli,
#         [
#             "cell-detection",
#             "--brainways-path",
#             project_path,
#         ],
#     )
#     assert result.exit_code == 0, result.output
#     assert Path(tmpdir / "image_scene0.csv").exists()


def test_create_excel(subject_path: Path, mock_atlas: BrainwaysAtlas, tmpdir):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "create-excel",
        ],
    )
    assert result.exit_code == 2
