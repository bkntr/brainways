from dataclasses import fields

import numpy as np


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
