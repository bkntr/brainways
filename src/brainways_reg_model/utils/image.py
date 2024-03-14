from __future__ import annotations

from typing import Union

import numpy as np
import torch


def normalize_min_max(image: Union[np.ndarray, torch.Tensor]):
    return (image - image.min()) / (image.max() - image.min())


def slice_contrast_values(
    slice_image: Union[np.ndarray, torch.Tensor], saturation: float = 0.001
):
    if isinstance(slice_image, torch.Tensor):
        hist, bin_edges = torch.histogram(slice_image.flatten(), bins=1024)
    else:
        hist, bin_edges = np.histogram(slice_image.flat, bins=1024)
    if hist[0] > hist[1]:
        hist = hist[1:]
        bin_edges = bin_edges[1:]
    if hist[-1] > hist[-2]:
        hist = hist[:-1]
        bin_edges = bin_edges[:-1]

    count_sum = sum(hist)
    count_max = count_sum * saturation
    count = count_max
    min_display = bin_edges[0]
    ind = 0
    while ind < len(hist) - 1:
        next_count = hist[ind]
        if count < next_count:
            bin_width = bin_edges[ind + 1] - bin_edges[ind]
            min_display = bin_edges[ind] + (count / next_count) * bin_width
            break
        count -= next_count
        ind += 1

    count = count_max
    max_display = bin_edges[-1]
    ind = len(hist) - 1
    while ind >= 0:
        next_count = hist[ind]
        if count < next_count:
            bin_width = bin_edges[ind + 1] - bin_edges[ind]
            max_display = bin_edges[ind + 1] - (count / next_count) * bin_width
            break
        count -= next_count
        ind -= 1
    return min_display, max_display


def normalize_contrast_qupath(slice_image: np.ndarray):
    min_val, max_val = slice_contrast_values(slice_image)
    if isinstance(slice_image, torch.Tensor):
        slice_image = torch.clip(slice_image, min_val, max_val)
    else:
        slice_image = np.clip(slice_image, min_val, max_val)
    slice_image = normalize_min_max(slice_image)
    return slice_image
