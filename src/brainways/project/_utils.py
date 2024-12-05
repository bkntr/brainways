import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import dacite
from packaging import version

import brainways
from brainways.project.info_classes import SliceInfo, SubjectFileFormat, SubjectInfo
from brainways.utils.io_utils.readers.qupath_reader import QupathReader


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
    if version.parse(project_version) < version.parse("0.1.4"):
        update_project_to_0_1_4(path)
    if version.parse(project_version) < version.parse("0.1.5"):
        update_project_to_0_1_5(path)
    if version.parse(project_version) < version.parse("0.1.7"):
        update_project_to_0_1_7(path)
    if version.parse(project_version) < version.parse("0.1.16"):
        update_project_to_0_1_16(path)

    with open(path) as f:
        new_project_version = json.load(f)["version"]

    if version.parse(new_project_version) < version.parse(brainways._version.version):
        rewrite_project_version(path)


def update_project_to_0_1_4(path: Path):
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


def update_project_to_0_1_5(path: Path):
    brainways_subject_paths = path.parent.rglob("brainways.bin")
    for brainways_subject_path in brainways_subject_paths:
        with open(brainways_subject_path, "rb") as f:
            serialized_settings, serialized_slice_infos = pickle.load(f)
        for serialized_slice_info in serialized_slice_infos:
            reader = QupathReader(serialized_slice_info["path"]["filename"])
            pps = reader.physical_pixel_sizes
            serialized_slice_info["physical_pixel_sizes"] = (pps.Y, pps.X)
        with open(brainways_subject_path, "wb") as f:
            pickle.dump((serialized_settings, serialized_slice_infos), f)

    rewrite_project_version(path=path, version="0.1.5")


def update_project_to_0_1_7(path: Path):
    brainways_subject_paths = path.parent.rglob("brainways.bin")
    for brainways_subject_path in brainways_subject_paths:
        with open(brainways_subject_path, "rb") as f:
            serialized_settings, serialized_slice_infos = pickle.load(f)
        subject_info = SubjectInfo(
            name=brainways_subject_path.parent.name,
            registration_channel=0,
            cell_detection_channels=[0],
        )
        slice_infos = []
        for serialized_slice_info in serialized_slice_infos:
            if (
                "tps" in serialized_slice_info["params"]
                and serialized_slice_info["params"]["tps"] is not None
            ):
                serialized_slice_info["params"]["tps"]["points_src"] = (
                    serialized_slice_info["params"]["tps"]["points_src"].tolist()
                )
                serialized_slice_info["params"]["tps"]["points_dst"] = (
                    serialized_slice_info["params"]["tps"]["points_dst"].tolist()
                )
                slice_infos.append(dacite.from_dict(SliceInfo, serialized_slice_info))
        subject_file = SubjectFileFormat(
            subject_info=subject_info,
            slice_infos=slice_infos,
        )
        with open(brainways_subject_path.parent / "data.bws", "w") as f:
            json.dump(asdict(subject_file), f)
        brainways_subject_path.unlink()

    rewrite_project_version(path=path, version="0.1.7")


def update_project_to_0_1_16(path: Path):
    with open(path) as f:
        serialized_project_settings = json.load(f)
    project_channel = serialized_project_settings["channel"]
    del serialized_project_settings["channel"]
    with open(path, "w") as f:
        json.dump(serialized_project_settings, f)
    brainways_subject_paths = path.parent.rglob("*.bws")
    for brainways_subject_path in brainways_subject_paths:
        with open(brainways_subject_path) as f:
            serialized_subject = json.load(f)
        serialized_subject["subject_info"]["registration_channel"] = project_channel
        serialized_subject["subject_info"]["cell_detection_channels"] = [
            project_channel
        ]
        with open(brainways_subject_path, "w") as f:
            json.dump(serialized_subject, f)

    rewrite_project_version(path=path, version="0.1.16")
