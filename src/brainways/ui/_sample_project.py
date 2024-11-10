import shutil
from pathlib import Path

import numpy as np
from datasets.download import DownloadManager

from brainways.utils.paths import get_brainways_dir


def _load_sample_project(annotated: bool):
    if annotated:
        project_name = "sample-project-annotated"
    else:
        project_name = "sample-project"

    download_path = DownloadManager().download_and_extract(
        f"https://huggingface.co/datasets/brainways/sample-project/resolve/main/{project_name}.zip"
    )
    project_path = get_brainways_dir() / project_name
    if project_path.exists():
        shutil.rmtree(project_path)
    shutil.copytree(Path(download_path) / project_name, project_path)
    layer_metadata = {
        "__brainways__": True,
        "sample_project_path": project_path / "brainways.bwp",
    }
    return [
        (
            np.zeros((10, 10)),
            {"metadata": layer_metadata, "visible": False, "name": "__helper__"},
        )
    ]


def load_sample_project():
    return _load_sample_project(annotated=False)


def load_sample_project_annotated():
    return _load_sample_project(annotated=True)
