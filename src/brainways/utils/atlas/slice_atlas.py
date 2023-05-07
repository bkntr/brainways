from typing import Tuple

import numpy as np
import torch.nn.functional
from numpy.typing import ArrayLike


def homog_indices(s0: int, s1: int, s2: int):
    idxs = np.mgrid[:s0, :s1, :s2, 1:2].reshape(-1, s0 * s1 * s2)
    return idxs.astype(np.int32)


def translate(idxs: ArrayLike, t0: float, t1: float, t2: float):
    return idxs + [[t0], [t1], [t2], [0]]


def rotate(idxs: ArrayLike, r0: float, r1: float, r2: float):
    r0, r1, r2 = np.radians([r0, r1, r2])
    rotmat_0 = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(r0), -np.sin(r0), 0],
            [0, np.sin(r0), np.cos(r0), 0],
            [0, 0, 0, 1],
        ]
    )

    rotmat_1 = np.array(
        [
            [np.cos(r1), 0, -np.sin(r1), 0],
            [0, 1, 0, 0],
            [np.sin(r1), 0, np.cos(r1), 0],
            [0, 0, 0, 1],
        ]
    )

    rotmat_2 = np.array(
        [
            [np.cos(r2), -np.sin(r2), 0, 0],
            [np.sin(r2), np.cos(r2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return rotmat_2 @ rotmat_1 @ rotmat_0 @ idxs


def homog_center_at_zero(idxs: ArrayLike):
    shape = idxs.max(axis=1)
    center = shape / 2
    return translate(idxs, -center[0], -center[1], -center[2])


def remap(array: torch.Tensor, indices: ArrayLike, mode="bilinear") -> torch.Tensor:
    assert array.dtype == torch.float32
    input = array[None, None, ...]  # NCDHW
    grid = torch.tensor(indices).flip(dims=[0]).T[None, None, None, ...]  # NDHW3
    grid = grid / (torch.tensor(array.shape[::-1]) - 1) * 2 - 1
    out = torch.nn.functional.grid_sample(
        input, grid.float(), mode=mode, align_corners=True
    )
    return out


def slice_atlas(
    shape: Tuple[int, int],
    volume: torch.Tensor,
    ap: float,
    si: float,
    lr: float,
    rot_frontal: float,
    rot_horizontal: float,
    rot_sagittal: float,
    interpolation: str = "bilinear",
):
    idxs = homog_indices(1, shape[0], shape[1])
    idxs = homog_center_at_zero(idxs)
    idxs = rotate(idxs, rot_frontal, rot_horizontal, rot_sagittal)
    idxs = translate(idxs, t0=ap, t1=si, t2=lr)
    idxs = idxs.astype(np.float32)
    slice = remap(volume, idxs[:3], interpolation)
    slice = slice.reshape(shape[0], shape[1])
    return slice
