from __future__ import annotations

from dataclasses import fields
from typing import Tuple

import numpy as np
from napari.layers import Image


def update_layer_contrast_limits(
    layer: Image,
    contrast_limits_quantiles: Tuple[float, float] = (0.01, 0.98),
    contrast_limits_range_quantiles: Tuple[float, float] = (0.0, 1.0),
) -> None:
    nonzero_mask = layer.data > 0
    if (~nonzero_mask).all():
        return

    limit_0, limit_1, limit_range_0, limit_range_1 = np.quantile(
        layer.data[nonzero_mask],
        (*contrast_limits_quantiles, *contrast_limits_range_quantiles),
    )
    layer.contrast_limits = (limit_0, limit_1 + 1e-8)
    layer.contrast_limits_range = (limit_range_0, limit_range_1 + 1e-8)


def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    try:
        return a == b
    except TypeError:
        return NotImplemented


def dataclass_eq(dc1, dc2) -> bool:
    """checks if two dataclasses which hold numpy arrays are equal"""
    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:
        return NotImplemented
    fields_names = [f.name for f in fields(dc1)]
    return all(
        array_safe_eq(getattr(dc1, field_name), getattr(dc2, field_name))
        for field_name in fields_names
    )
