import json
import logging
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, Union

import dacite
import numpy as np
import pandas as pd
from PIL import Image

from brainways.pipeline.brainways_params import (
    AtlasRegistrationParams,
    CellDetectorParams,
)
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline, PipelineStep
from brainways.pipeline.cell_detector import CellDetector
from brainways.project.info_classes import (
    ExcelMode,
    MaskFileFormat,
    SliceInfo,
    SubjectFileFormat,
    SubjectInfo,
)
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cell_count_summary import cell_count_summary
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from brainways.utils.cells import (
    filter_cells_by_size,
    filter_cells_on_annotation,
    filter_cells_on_tissue,
    get_region_areas,
)
from brainways.utils.export import export_mask
from brainways.utils.image import brain_mask_simple, get_resize_size, slice_to_uint8
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers import get_image_size
from brainways.utils.io_utils.readers.qupath_reader import QupathReader

if TYPE_CHECKING:
    from brainways.project.brainways_project import BrainwaysProject


class BrainwaysSubject:
    def __init__(
        self,
        subject_info: SubjectInfo,
        slice_infos: List[SliceInfo],
        project: "BrainwaysProject",
    ):
        self.subject_info = subject_info
        self.documents = slice_infos
        self.project = project
        self._save_dir = project.path.parent / self.subject_info.name

    @classmethod
    def create(
        cls, subject_info: SubjectInfo, project: "BrainwaysProject"
    ) -> "BrainwaysSubject":
        subject = cls(subject_info=subject_info, slice_infos=[], project=project)

        if subject._save_dir.exists():
            raise FileExistsError(
                f"Subject directory {subject._save_dir} already exists"
            )

        subject._save_dir.mkdir()
        subject.thumbnails_root.mkdir()
        subject.cell_detections_root.mkdir()
        subject.save()
        return subject

    @classmethod
    def open(cls, path: Union[Path, str], project: "BrainwaysProject"):
        if not path.exists():
            raise FileNotFoundError(f"Subject file not found: {path}")

        with open(path) as f:
            serialized_file = json.load(f)

        subject_file = dacite.from_dict(
            SubjectFileFormat, serialized_file, config=dacite.Config(cast=[tuple])
        )
        subject = BrainwaysSubject(
            subject_info=subject_file.subject_info,
            slice_infos=subject_file.slice_infos,
            project=project,
        )

        return subject

    def read_lowres_image(
        self, document: SliceInfo, channel: Optional[int] = None
    ) -> np.ndarray:
        thumbnail_path = self.thumbnail_path(
            document.path, channel=channel or self.project.settings.channel
        )
        if thumbnail_path.exists():
            image = np.array(Image.open(thumbnail_path))
        else:
            reader = QupathReader(document.path.filename)
            reader.set_scene(document.path.scene)
            image = reader.get_thumbnail(
                target_size=document.lowres_image_size,
                channel=channel or self.project.settings.channel,
            )
            image = slice_to_uint8(image)
            Image.fromarray(image).save(thumbnail_path)
        return image

    def read_highres_image(self, document: SliceInfo, level: Optional[int] = None):
        reader = QupathReader(document.path.filename)
        reader.set_scene(document.path.scene)
        if level:
            reader.set_level(level)
        image = reader.get_image_dask_data(
            "YX", C=self.project.settings.channel
        ).compute()
        return image

    def add_image(self, path: ImagePath, load_thumbnail: bool = True) -> SliceInfo:
        image_size = get_image_size(path)
        lowres_image_size = get_resize_size(
            input_size=image_size, output_size=(1024, 1024), keep_aspect=True
        )
        pps = QupathReader(path.filename).physical_pixel_sizes
        document = SliceInfo(
            path=path,
            image_size=image_size,
            lowres_image_size=lowres_image_size,
            physical_pixel_sizes=(pps.Y, pps.X),
        )
        if load_thumbnail:
            self.read_lowres_image(document)
        self.documents.append(document)
        return document

    def save(self):
        subject_file = SubjectFileFormat(
            subject_info=self.subject_info, slice_infos=self.documents
        )
        serialized_file = asdict(subject_file)
        with open(self._save_dir / "data.bws", "w") as f:
            json.dump(serialized_file, f)

    def move_images_root(
        self,
        new_images_root: Union[Path, str],
        old_images_root: Optional[Union[Path, str]] = None,
    ):
        new_images_root = Path(new_images_root)
        for i, document in enumerate(self.documents):
            cur_filename = Path(document.path.filename)
            if old_images_root is None:
                cur_old_images_root = cur_filename.parent
            else:
                cur_old_images_root = Path(old_images_root)
            cur_relative_filename = cur_filename.relative_to(cur_old_images_root)
            new_filename = new_images_root / cur_relative_filename
            if not new_filename.exists():
                logging.warning(
                    f"{new_filename} not found, skipping! (old filename {cur_filename})"
                )
                continue
            new_path = replace(document.path, filename=str(new_filename))
            self.documents[i] = replace(document, path=new_path)

    def read_cell_detections(self, document: SliceInfo):
        return pd.read_csv(self.cell_detections_path(document.path))

    def import_cell_detections_iter(
        self,
        root: Path,
        cell_detection_importer: CellDetectionImporter,
    ) -> Iterator:
        for i, document in self.valid_documents:
            cell_detections_path = cell_detection_importer.find_cell_detections_file(
                root=root, document=document
            )
            if cell_detections_path is None:
                logging.warning(f"found no cells for document: '{document.path}'")
                continue

            cells_df = cell_detection_importer.read_cells_file(
                path=cell_detections_path, document=document
            )
            cells_df.to_csv(self.cell_detections_path(document.path), index=False)
            yield

    def import_cell_detections(
        self,
        root: Path,
        cell_detection_importer: CellDetectionImporter,
    ) -> None:
        for _ in self.import_cell_detections_iter(
            root=root, cell_detection_importer=cell_detection_importer
        ):
            pass

    def run_cell_detector(
        self,
        slice_info: SliceInfo,
        cell_detector: CellDetector,
        default_params: CellDetectorParams,
        save_cell_detection_masks_file_format: Optional[MaskFileFormat] = None,
    ) -> None:
        cell_detections_path = self.cell_detections_path(slice_info.path)
        cell_detections_path.parent.mkdir(parents=True, exist_ok=True)
        reader = slice_info.image_reader()
        image = reader.get_image_dask_data(
            "YX", C=self.project.settings.channel
        ).compute()
        if slice_info.params.cell is not None:
            cell_detector_params = slice_info.params.cell
        else:
            cell_detector_params = default_params
        labels = cell_detector.run_cell_detector(
            image,
            params=cell_detector_params,
            physical_pixel_sizes=slice_info.physical_pixel_sizes,
        )
        cells = cell_detector.cells(
            labels=labels,
            image=image,
            physical_pixel_sizes=slice_info.physical_pixel_sizes,
        )
        cells.to_csv(cell_detections_path, index=False)

        if save_cell_detection_masks_file_format is not None:
            mask_path = (
                self.project.path.parent
                / "__outputs__"
                / "cell_detection_masks"
                / self.subject_info.name
                / Path(str(slice_info.path)).name
            )
            export_mask(
                data=labels,
                path=mask_path,
                file_format=save_cell_detection_masks_file_format,
            )

    def clear_cell_detection(self, slice_info: SliceInfo):
        cell_detections_path = self.cell_detections_path(slice_info.path)
        if cell_detections_path.exists():
            cell_detections_path.unlink()

    def get_valid_cells(
        self, document: SliceInfo, annotation: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        if annotation is None:
            atlas_slice = self.pipeline.get_atlas_slice(document.params)
            annotation = atlas_slice.annotation.numpy()

        image = self.read_lowres_image(document)
        cells = self.read_cell_detections(document)
        valid_cells = filter_cells_on_tissue(cells=cells, image=image)
        valid_cells = filter_cells_on_annotation(
            cells=valid_cells,
            lowres_image_size=document.lowres_image_size,
            params=document.params,
            pipeline=self.pipeline,
            annotation=annotation,
        )
        return valid_cells

    def get_cells_on_atlas(
        self, documents: Optional[List[SliceInfo]] = None
    ) -> Optional[pd.DataFrame]:
        all_cells_on_atlas = []
        if documents is None:
            documents = (document for i, document in self.valid_documents)
        for document in documents:
            if not self.cell_detections_path(document.path).exists():
                logging.warning(
                    f"{document.path}: missing cells, please run cell detection."
                )
                continue
            image_to_atlas_transform = self.pipeline.get_image_to_atlas_transform(
                brainways_params=document.params,
                lowres_image_size=document.lowres_image_size,
            )
            cells = self.get_valid_cells(document)
            cells_on_image = cells[["x", "y"]].values * document.lowres_image_size[::-1]
            cells_on_atlas = image_to_atlas_transform.transform_points(cells_on_image)
            cells.loc[:, ["x", "y", "z"]] = cells_on_atlas
            all_cells_on_atlas.append(cells)

        if len(all_cells_on_atlas) == 0:
            return None

        all_cells_on_atlas = pd.concat(all_cells_on_atlas, axis=0)
        return all_cells_on_atlas

    def cell_count_summary(
        self,
        slice_info_predicate: Optional[Callable[[SliceInfo], bool]] = None,
        min_region_area_um2: Optional[int] = None,
        cells_per_area_um2: Optional[int] = None,
        min_cell_size_um: Optional[float] = None,
        max_cell_size_um: Optional[float] = None,
        excel_mode: ExcelMode = ExcelMode.ROW_PER_SUBJECT,
    ):
        if self.pipeline is None:
            raise RuntimeError(
                "BrainwaysPipeline not loaded, run BrainwaysProject.load_pipeline()"
            )

        all_region_areas = Counter()
        all_cells_on_atlas = []
        image_dfs = []
        for _, document in self.valid_documents:
            if (
                slice_info_predicate is not None
                and slice_info_predicate(document) is False
            ):
                continue
            if not self.cell_detections_path(document.path).exists():
                logging.warning(
                    f"{document.path}: missing cells, please run cell detection."
                )
                continue
            image = self.read_lowres_image(document)
            image_to_atlas_transform = self.pipeline.get_image_to_atlas_transform(
                brainways_params=document.params,
                lowres_image_size=document.lowres_image_size,
            )
            image_to_atlas_slice_transform = self.pipeline.get_image_to_atlas_transform(
                brainways_params=document.params,
                lowres_image_size=document.lowres_image_size,
                until_step=PipelineStep.TPS,
            )
            atlas_slice = self.pipeline.get_atlas_slice(document.params)
            annotation = atlas_slice.annotation.numpy()
            cells = self.get_valid_cells(document, annotation=annotation)
            cells = filter_cells_by_size(
                cells, min_size_um=min_cell_size_um, max_size_um=max_cell_size_um
            )
            cells_on_image = cells[["x", "y"]].values * document.lowres_image_size[::-1]
            registered_image = image_to_atlas_slice_transform.transform_image(
                image,
                output_size=atlas_slice.shape,
            )
            cells_on_atlas = image_to_atlas_transform.transform_points(cells_on_image)
            cells_on_registered_image = image_to_atlas_slice_transform.transform_points(
                cells_on_image
            )

            brain_mask = brain_mask_simple(registered_image)
            region_areas = get_region_areas(
                annotation=annotation,
                atlas=self.atlas,
                mask=brain_mask,
            )
            cells.loc[:, "x"] = cells_on_atlas[:, 0]
            cells.loc[:, "y"] = cells_on_atlas[:, 1]
            cells.loc[:, "z"] = cells_on_atlas[:, 2]

            cells_on_registered_image_int = cells_on_registered_image.round().astype(
                np.int32
            )
            cells.loc[:, "struct_id"] = annotation[
                cells_on_registered_image_int[:, 1], cells_on_registered_image_int[:, 0]
            ]

            if excel_mode == ExcelMode.ROW_PER_IMAGE:
                image_df = cell_count_summary(
                    animal_id=self.subject_info.name,
                    cells=cells,
                    region_areas_um=region_areas,
                    atlas=self.atlas,
                    min_region_area_um2=min_region_area_um2,
                    cells_per_area_um2=cells_per_area_um2,
                    conditions=self.subject_info.conditions,
                )
                image_df["image_path"] = str(document.path)
                image_df["ap (µm)"] = (
                    document.params.atlas.ap
                    * self.pipeline.atlas.brainglobe_atlas.resolution[0]
                )
                image_dfs.append(image_df)
            else:
                all_cells_on_atlas.append(cells)
                all_region_areas.update(region_areas)

        if excel_mode == ExcelMode.ROW_PER_IMAGE:
            if len(image_dfs) == 0:
                logging.warning(f"{self.subject_info.name}: not found cells on atlas")
                return

            df = pd.concat(image_dfs, axis=0)
        else:
            if len(all_cells_on_atlas) == 0:
                logging.warning(f"{self.subject_info.name}: not found cells on atlas")
                return

            all_cells_on_atlas = pd.concat(all_cells_on_atlas, axis=0)
            df = cell_count_summary(
                animal_id=self.subject_info.name,
                cells=all_cells_on_atlas,
                region_areas_um=all_region_areas,
                atlas=self.atlas,
                min_region_area_um2=min_region_area_um2,
                cells_per_area_um2=cells_per_area_um2,
                conditions=self.subject_info.conditions,
            )
        return df

    def thumbnail_path(self, image_path: ImagePath, channel: Optional[int] = None):
        if channel is None:
            channel = self.project.settings.channel

        suffixes = []
        if image_path.scene is not None:
            suffixes.append(f"Scene #{image_path.scene}")
        suffixes.append(f"Channel #{channel}")
        suffix = " ".join(suffixes)
        thumbnail_filename = f"{Path(image_path.filename).stem} [{suffix}].jpg"
        return self.thumbnails_root / thumbnail_filename

    def cell_detections_path(self, image_path: ImagePath) -> Path:
        return self.cell_detections_root / (Path(str(image_path)).name + ".csv")

    def set_rotation(self, rot_horizontal: float, rot_sagittal: float):
        self.subject_info = replace(
            self.subject_info,
            rotation=(rot_horizontal, rot_sagittal),
        )
        for i, document in enumerate(self.documents):
            atlas_params = document.params.atlas
            if atlas_params is not None:
                atlas_params = replace(
                    atlas_params,
                    rot_horizontal=rot_horizontal,
                    rot_sagittal=rot_sagittal,
                )
                self.documents[i] = replace(
                    document, params=replace(document.params, atlas=atlas_params)
                )

    def evenly_space_slices_on_ap_axis(self):
        if len(self.valid_documents) <= 2:
            return

        _, first_slice_info = self.valid_documents[0]
        _, last_slice_info = self.valid_documents[-1]

        if (
            first_slice_info.params.atlas is None
            or last_slice_info.params.atlas is None
        ):
            raise ValueError(
                "First and last slices must have atlas parameters, please go to the first and last slices and select an AP value."
            )

        first_ap = first_slice_info.params.atlas.ap
        last_ap = last_slice_info.params.atlas.ap
        ap_diff = last_ap - first_ap
        ap_step = ap_diff / (len(self.valid_documents) - 1)

        for valid_document_index, (document_index, document) in enumerate(
            self.valid_documents
        ):
            atlas_params = document.params.atlas
            if atlas_params is None:
                atlas_params = AtlasRegistrationParams()
                # Set the rotation parameters
                if self.subject_info.rotation is not None:
                    atlas_params = replace(
                        atlas_params,
                        rot_horizontal=self.subject_info.rotation[0],
                        rot_sagittal=self.subject_info.rotation[1],
                    )

            atlas_params = replace(
                atlas_params, ap=first_ap + valid_document_index * ap_step
            )
            self.documents[document_index] = replace(
                document, params=replace(document.params, atlas=atlas_params)
            )

    @property
    def thumbnails_root(self) -> Path:
        return self._save_dir / "thumbnails"

    @property
    def cell_detections_root(self) -> Path:
        return self._save_dir / "cell_detections"

    @property
    def valid_documents(self) -> List[Tuple[int, SliceInfo]]:
        return [
            (i, document)
            for i, document in enumerate(self.documents)
            if not document.ignore
        ]

    @property
    def pipeline(self) -> BrainwaysPipeline:
        return self.project.pipeline

    @property
    def atlas(self) -> BrainwaysAtlas:
        return self.project.atlas
