import logging
from functools import cached_property
from pathlib import Path
from typing import Tuple

import cv2
import dask.array as da
import numpy as np
import pandas as pd
from csbdeep.data import Normalizer
from skimage.measure import regionprops_table
from stardist.models import StarDist2D

from brainways.pipeline.brainways_params import CellDetectorParams
from brainways.utils.image import normalize_contrast


class ClaheNormalizer(Normalizer):
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    def before(self, x, axes):
        if isinstance(x, da.Array):
            x = x.compute()
        x = self.clahe.apply(x.astype(np.uint16))
        x = x.astype(np.float32)
        x = normalize_contrast(x, 0.0, 99.8)
        x = x.squeeze()
        return x[..., None]

    def after(self, mean, scale, axes):
        return mean, scale

    @property
    def do_after(self):
        return False


class MinMaxNormalizer(Normalizer):
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def before(self, x, axes):
        x = np.clip(x, self.min_value, self.max_value)
        x = (x - self.min_value) / (self.max_value - self.min_value)
        x = x.astype(np.float32)
        x = x.squeeze()
        return x[..., None]

    def after(self, mean, scale, axes):
        return mean, scale

    @property
    def do_after(self):
        return False


class QuantileNormalizer(Normalizer):
    def __init__(self, min_quantile: float, max_quantile: float):
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

    def before(self, x, axes):
        min_value, max_value = np.quantile(x, [self.min_quantile, self.max_quantile])
        x = (x - min_value) / (max_value - min_value)
        x = np.clip(x, 0, 1)
        x = x.astype(np.float32)
        x = x.squeeze()
        return x[..., None]

    def after(self, mean, scale, axes):
        return mean, scale

    @property
    def do_after(self):
        return False


def filter_by_cell_size(
    labels: np.ndarray,
    image: np.ndarray,
    params: CellDetectorParams,
    physical_pixel_sizes: Tuple[float, float],
):
    regionprops = pd.DataFrame(
        regionprops_table(
            labels,
            image,
            properties=("label", "area"),
        )
    )
    if np.isnan(physical_pixel_sizes).any():
        logging.warning(
            "Images do not have pixel size information, filtering cell size by pixels."
        )
    else:
        regionprops["area"] = (
            regionprops["area"] * physical_pixel_sizes[0] * physical_pixel_sizes[1]
        )

    regionprops["include"] = True
    if params.cell_size_range[0] > 0:
        regionprops["include"] &= regionprops["area"] >= params.cell_size_range[0]
    if params.cell_size_range[1] > 0:
        regionprops["include"] &= regionprops["area"] <= params.cell_size_range[1]

    labels_include = np.in1d(
        labels.flat, regionprops[regionprops["include"]]["label"]
    ).reshape(labels.shape)
    return labels * labels_include.astype(labels.dtype)


class CellDetector:
    def __init__(self, custom_model_dir: str = ""):
        self.custom_model_dir = custom_model_dir

    @cached_property
    def stardist(self):
        if self.custom_model_dir:
            logging.info(f"Using custom StarDist model from {self.custom_model_dir}")
            return StarDist2D(
                None,
                name=Path(self.custom_model_dir).name,
                basedir=Path(self.custom_model_dir).parent,
            )
        else:
            logging.info("Using default StarDist model")
            return StarDist2D.from_pretrained("2D_versatile_fluo")

    def _filter_detections_by_area(self, labels: np.ndarray):
        regionprops_df = pd.DataFrame(
            regionprops_table(labels, properties=("label", "centroid", "area"))
        )

        cell_area = regionprops_df["area"]
        cell_area_mean = cell_area.mean()
        cell_area_std = cell_area.std()
        bad_cell_rows = (cell_area < cell_area_mean - 3 * cell_area_std) | (
            cell_area > cell_area_mean + 3 * cell_area_std
        )
        bad_label_idxs = regionprops_df[bad_cell_rows]["label"].to_numpy()
        bad_label_mask = (labels[..., None] == bad_label_idxs).any(axis=-1)
        labels[bad_label_mask] = 0
        return labels

    def run_cell_detector(
        self,
        image,
        params: CellDetectorParams,
        physical_pixel_sizes: Tuple[float, float],
        block_size: int = 2048,
        **kwargs,
    ) -> np.ndarray:
        normalizer = self.get_normalizer(params)
        labels, _ = self.predict_cells(image, normalizer, block_size, **kwargs)

        if params.cell_size_range != (0, 0):
            labels = filter_by_cell_size(
                labels=labels,
                image=image,
                params=params,
                physical_pixel_sizes=physical_pixel_sizes,
            )

        return labels

    def cells(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        physical_pixel_sizes: Tuple[float, float],
    ):
        regionprops_df = pd.DataFrame(
            regionprops_table(
                labels, image, properties=("centroid", "area", "mean_intensity")
            )
        ).astype(int)
        df = pd.DataFrame()
        df["x"] = regionprops_df["centroid-1"] / image.shape[1]
        df["y"] = regionprops_df["centroid-0"] / image.shape[0]
        df["area_pixels"] = regionprops_df["area"]
        df["area_um"] = (
            regionprops_df["area"] * physical_pixel_sizes[0] * physical_pixel_sizes[1]
        )
        df["mean_intensity"] = regionprops_df["mean_intensity"]

        return df

    def get_normalizer(self, params: CellDetectorParams):
        if params.normalizer == "quantile":
            normalizer = QuantileNormalizer(
                min_quantile=params.normalizer_range[0],
                max_quantile=params.normalizer_range[1],
            )
        elif params.normalizer == "value":
            normalizer = MinMaxNormalizer(
                min_value=params.normalizer_range[0],
                max_value=params.normalizer_range[1],
            )
        elif params.normalizer == "clahe":
            normalizer = ClaheNormalizer()
        elif params.normalizer == "none":
            normalizer = None
        else:
            raise ValueError(f"Unknown normalizer {params.normalizer}.")

        return normalizer

    def predict_cells(
        self, image: np.ndarray, normalizer: Normalizer, block_size: int, **kwargs
    ):
        image_size = min(image.shape[:2])
        if image_size > block_size:
            return self.stardist.predict_instances_big(
                image,
                axes="YX",
                block_size=block_size,
                min_overlap=block_size // 16,
                normalizer=normalizer,
                **kwargs,
            )
        else:
            return self.stardist.predict_instances(
                image,
                axes="YX",
                normalizer=normalizer,
                **kwargs,
            )
