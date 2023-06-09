import logging
import pickle
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import dacite
import numpy as np
import pandas as pd
from PIL import Image

from brainways.pipeline.brainways_params import CellDetectorParams
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline, PipelineStep
from brainways.pipeline.cell_detector import CellDetector
from brainways.project.info_classes import ExcelMode, ProjectSettings, SliceInfo
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
from brainways.utils.image import brain_mask_simple, get_resize_size, slice_to_uint8
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers import get_image_size
from brainways.utils.io_utils.readers.qupath_reader import QupathReader


class BrainwaysSubject:
    def __init__(
        self,
        settings: ProjectSettings,
        documents: List[SliceInfo] = None,
        subject_path: Optional[Union[Path, str]] = None,
        atlas: Optional[BrainwaysAtlas] = None,
        pipeline: Optional[BrainwaysPipeline] = None,
    ):
        if atlas is not None:
            if atlas.brainglobe_atlas.atlas_name != settings.atlas:
                raise ValueError(
                    "Input atlas doesn't match atlas in subject settings "
                    f"({atlas.brainglobe_atlas.atlas_name} != {settings.atlas})"
                )
        self.documents: List[SliceInfo] = documents or []
        self.settings = settings
        self.atlas = atlas
        self.pipeline = pipeline
        self._tmpdir = None

        # TODO: refactor, BrainwaysSubject.create() and BrainwaysSubject.open()
        if subject_path is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.subject_path = Path(self._tmpdir.name)
        else:
            self.subject_path = self._get_subject_dir(subject_path)
            if not (self.subject_path / "brainways.bin").exists():
                if self.subject_path.exists():
                    if not self.subject_path.is_dir() or any(
                        self.subject_path.iterdir()
                    ):
                        raise FileExistsError(
                            f"New subject directory {self.subject_path} is not empty!"
                        )
                else:
                    self.subject_path.mkdir()

        if not self.thumbnails_root.exists():
            self.thumbnails_root.mkdir()

        if not self.cell_detections_root.exists():
            self.cell_detections_root.mkdir()

    def close(self):
        self.documents = []
        self.settings = None
        self.atlas = None
        self.pipeline = None
        self.subject_path = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def read_lowres_image(
        self, document: SliceInfo, channel: Optional[int] = None
    ) -> np.ndarray:
        thumbnail_path = self.thumbnail_path(
            document.path, channel=channel or self.settings.channel
        )
        if thumbnail_path.exists():
            image = np.array(Image.open(thumbnail_path))
        else:
            reader = QupathReader(document.path.filename)
            reader.set_scene(document.path.scene)
            image = reader.get_thumbnail(
                target_size=document.lowres_image_size,
                channel=channel or self.settings.channel,
            )
            image = slice_to_uint8(image)
            Image.fromarray(image).save(thumbnail_path)
        return image

    def read_highres_image(self, document: SliceInfo, level: Optional[int] = None):
        reader = QupathReader(document.path.filename)
        reader.set_scene(document.path.scene)
        if level:
            reader.set_level(level)
        image = reader.get_image_dask_data("YX", C=self.settings.channel).compute()
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

    @staticmethod
    def _get_subject_dir(path: Union[Path, str]):
        subject_dir = Path(path)
        if subject_dir.name == "brainways.bin":
            subject_dir = subject_dir.parent
        return subject_dir

    @classmethod
    def open(
        cls,
        path: Union[Path, str],
        atlas: Optional[BrainwaysAtlas] = None,
        pipeline: Optional[BrainwaysPipeline] = None,
    ):
        subject_dir = BrainwaysSubject._get_subject_dir(path)
        if not subject_dir.exists():
            raise FileNotFoundError(f"subject path not found: {path}")

        with open(subject_dir / "brainways.bin", "rb") as f:
            serialized_settings, serialized_documents = pickle.load(f)

        settings = dacite.from_dict(ProjectSettings, serialized_settings)
        documents = [dacite.from_dict(SliceInfo, d) for d in serialized_documents]
        subject = BrainwaysSubject(
            settings=settings,
            documents=documents,
            subject_path=subject_dir,
            atlas=atlas,
            pipeline=pipeline,
        )

        return subject

    def save(self, path: Optional[Union[Path, str]] = None):
        if path is None:
            path = self.subject_path
        path = Path(path)
        subject_dir = self._get_subject_dir(path)
        if subject_dir != self.subject_path:
            if subject_dir.exists():
                if subject_dir.is_dir() and not any(subject_dir.iterdir()):
                    shutil.rmtree(str(subject_dir))
                else:
                    raise FileExistsError(
                        f"subject directory {subject_dir} is not empty!"
                    )
            shutil.move(str(self.subject_path), str(subject_dir))
            self.subject_path = subject_dir
        serialized_settings = asdict(self.settings)
        serialized_documents = [asdict(d) for d in self.documents]
        with open(subject_dir / "brainways.bin", "wb") as f:
            pickle.dump((serialized_settings, serialized_documents), f)

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

    def run_cell_detector_iter(
        self, cell_detector: CellDetector, default_params: CellDetectorParams
    ) -> Iterator:
        for i, document in self.valid_documents:
            try:
                cell_detections_path = self.cell_detections_path(document.path)
                if cell_detections_path.exists():
                    continue
                reader = document.image_reader()
                image = reader.get_image_dask_data(
                    "YX", C=self.settings.channel
                ).compute()
                if document.params.cell is not None:
                    cell_detector_params = document.params.cell
                else:
                    cell_detector_params = default_params
                labels = cell_detector.run_cell_detector(
                    image, params=cell_detector_params
                )
                cells = cell_detector.cells(
                    labels=labels,
                    image=image,
                    physical_pixel_sizes=document.physical_pixel_sizes,
                )
                cells.to_csv(cell_detections_path, index=False)
            except Exception:
                logging.exception(f"Cell detector on {document.path}")
            yield

    def run_cell_detector(
        self, cell_detector: CellDetector, default_params: CellDetectorParams
    ) -> None:
        for _ in self.run_cell_detector_iter(
            cell_detector=cell_detector, default_params=default_params
        ):
            pass

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
            document: SliceInfo
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
                # raise RuntimeError(
                #     f"{document.path}: missing cells, please run cell detection."
                # )
            if (
                document.params is None
                or document.params.atlas is None
                or document.params.affine is None
                or document.params.tps is None
            ):
                logging.warning(f"{document.path}: missing params.")
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
                image_dfs.append(
                    cell_count_summary(
                        animal_id=str(document.path),
                        cells=cells,
                        region_areas_um=region_areas,
                        atlas=self.atlas,
                        min_region_area_um2=min_region_area_um2,
                        cells_per_area_um2=cells_per_area_um2,
                    )
                )
            else:
                all_cells_on_atlas.append(cells)
                all_region_areas.update(region_areas)

        if excel_mode == ExcelMode.ROW_PER_IMAGE:
            df = pd.concat(image_dfs, axis=0)
        else:
            if len(all_cells_on_atlas) == 0:
                logging.warning(f"{document.path}: not found cells on atlas")
                return

            all_cells_on_atlas = pd.concat(all_cells_on_atlas, axis=0)
            df = cell_count_summary(
                animal_id=self.subject_path.name,
                cells=all_cells_on_atlas,
                region_areas_um=all_region_areas,
                atlas=self.atlas,
                min_region_area_um2=min_region_area_um2,
                cells_per_area_um2=cells_per_area_um2,
            )
        return df

    def thumbnail_path(self, image_path: ImagePath, channel: Optional[int] = None):
        if channel is None:
            channel = self.settings.channel

        suffixes = []
        if image_path.scene is not None:
            suffixes.append(f"Scene #{image_path.scene}")
        suffixes.append(f"Channel #{channel}")
        suffix = " ".join(suffixes)
        thumbnail_filename = f"{Path(image_path.filename).stem} [{suffix}].jpg"
        return self.thumbnails_root / thumbnail_filename

    def cell_detections_path(self, image_path: ImagePath) -> Path:
        return self.cell_detections_root / (Path(str(image_path)).name + ".csv")

    @property
    def thumbnails_root(self) -> Path:
        return self.subject_path / "thumbnails"

    @property
    def cell_detections_root(self) -> Path:
        return self.subject_path / "cell_detections"

    @property
    def valid_documents(self) -> List[Tuple[int, SliceInfo]]:
        return [
            (i, document)
            for i, document in enumerate(self.documents)
            if not document.ignore
        ]
