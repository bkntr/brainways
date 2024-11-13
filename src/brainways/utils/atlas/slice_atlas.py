from typing import Tuple

import numpy as np
import torch.nn.functional
from numpy.typing import NDArray


def homog_indices(s0: int, s1: int, s2: int) -> NDArray:
    idxs = np.mgrid[:s0, :s1, :s2, 1:2].reshape(-1, s0 * s1 * s2)
    return idxs.astype(np.int32)


def translate(idxs: NDArray, t0: float, t1: float, t2: float) -> NDArray:
    return idxs + [[t0], [t1], [t2], [0]]


def rotate(idxs: NDArray, r0: float, r1: float, r2: float) -> NDArray:
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


def homog_center_at_zero(idxs: NDArray) -> NDArray:
    shape = idxs.max(axis=1)
    center = shape / 2
    return translate(idxs, -center[0], -center[1], -center[2])


def remap(array: torch.Tensor, indices: NDArray, mode="bilinear") -> torch.Tensor:
    assert array.dtype == torch.float32
    input = array[None, None, ...]  # NCDHW
    grid = torch.tensor(indices).flip(dims=[0]).T[None, None, None, ...]  # NDHW3
    grid = grid / (torch.tensor(array.shape[::-1]) - 1) * 2 - 1
    out = torch.nn.functional.grid_sample(
        input, grid.float(), mode=mode, align_corners=True
    )
    return out


def get_slice_coordinates(
    shape: Tuple[int, int],
    ap: float,
    si: float,
    lr: float,
    rot_frontal: float,
    rot_horizontal: float,
    rot_sagittal: float,
    as_image: bool = True,
) -> NDArray:
    """
    Calculate the coordinates of a slice in a 3D space after applying rotations and translations.

    Args:
        shape (Tuple[int, int]): The shape of the slice (height, width).
        ap (float): The translation along the anterior-posterior axis.
        si (float): The translation along the superior-inferior axis.
        lr (float): The translation along the left-right axis.
        rot_frontal (float): The rotation angle around the frontal axis (in radians).
        rot_horizontal (float): The rotation angle around the horizontal axis (in radians).
        rot_sagittal (float): The rotation angle around the sagittal axis (in radians).
        as_image (bool): If True, the output will be a numpy array of shape (height, width, 3) containing the coordinates of the
            slice. If False, the output will be a numpy array of shape (3, height * width) containing the coordinates of the slice.

    Returns:
        NDArray: The coordinates of the slice. See `as_image` parameter for the shape of the output.
    """
    idxs = homog_indices(1, shape[0], shape[1])
    idxs = homog_center_at_zero(idxs)
    idxs = rotate(idxs, rot_frontal, rot_horizontal, rot_sagittal)
    idxs = translate(idxs, t0=ap, t1=si, t2=lr)
    idxs = idxs[:3].astype(np.float32)
    if as_image:
        idxs = idxs.transpose().reshape(shape[0], shape[1], 3)
    return idxs


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
    idxs = get_slice_coordinates(
        shape=shape,
        ap=ap,
        si=si,
        lr=lr,
        rot_frontal=rot_frontal,
        rot_horizontal=rot_horizontal,
        rot_sagittal=rot_sagittal,
        as_image=False,
    )
    slice = remap(volume, idxs[:3], interpolation)
    slice = slice.reshape(shape[0], shape[1])
    return slice
