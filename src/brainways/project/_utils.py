import json
import pickle
from pathlib import Path
from typing import Optional

from packaging import version

import brainways


def rewrite_project_version(path: Path, version: Optional[str] = None):
    if version is None:
        version = brainways._version.version

    with open(path) as f:
        serialized_project_settings = json.load(f)
    serialized_project_settings["version"] = version
    with open(path, "w") as f:
        json.dump(serialized_project_settings, f)


def update_project_from_previous_versions(path: Path):
    with open(path) as f:
        serialized_settings = json.load(f)
    project_version = serialized_settings.get("version", "0.1.1")
    if version.parse(project_version) <= version.parse("0.1.1"):
        update_project_from_0_1_1_to_0_1_4(path)

    rewrite_project_version(path)


def update_project_from_0_1_1_to_0_1_4(path: Path):
    brainways_subject_paths = path.parent.rglob("brainways.bin")
    for brainways_subject_path in brainways_subject_paths:
        with open(brainways_subject_path, "rb") as f:
            serialized_settings, serialized_slice_infos = pickle.load(f)
        for serialized_slice_info in serialized_slice_infos:
            if "cell" in serialized_slice_info["params"]:
                del serialized_slice_info["params"]["cell"]
        with open(brainways_subject_path, "wb") as f:
            pickle.dump((serialized_settings, serialized_slice_infos), f)

    rewrite_project_version(path=path, version="0.1.4")
