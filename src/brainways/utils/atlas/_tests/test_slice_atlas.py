import numpy as np
import numpy.testing as npt
import torch

from brainways.utils.atlas.slice_atlas import (
    homog_center_at_zero,
    homog_indices,
    rotate,
    slice_atlas,
    translate,
)


def test_homog_indices():
    idxs = homog_indices(1, 2, 2)
    expected = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]]
    npt.assert_allclose(idxs, expected)


def test_translate():
    idxs = np.array([[0, 1], [0, 1], [0, 1], [1, 1]])
    translated = translate(idxs, t0=-0.5, t1=-2, t2=-3)
    expected = [[-0.5, 0.5], [-2, -1], [-3, -2], [1, 1]]
    npt.assert_allclose(translated, expected)


def test_rotate_around_0():
    idxs = np.array([[0], [1], [0], [1]])
    rotated = rotate(idxs, r0=90, r1=0, r2=0)
    expected = np.array([[0], [0], [1], [1]])
    npt.assert_allclose(rotated, expected, atol=1e-7)


def test_rotate_around_1():
    idxs = np.array([[1], [0], [0], [1]])
    rotated = rotate(idxs, r0=0, r1=90, r2=0)
    expected = np.array([[0], [0], [1], [1]])
    npt.assert_allclose(rotated, expected, atol=1e-7)


def test_rotate_around_2():
    idxs = np.array([[1], [0], [0], [1]])
    rotated = rotate(idxs, r0=0, r1=0, r2=90)
    expected = np.array([[0], [1], [0], [1]])
    npt.assert_allclose(rotated, expected, atol=1e-7)


def test_homog_center_at_zero():
    idxs = np.array([[0, 1, 2], [0, 1, 2], [0, 0, 0], [1, 1, 1]])
    idxs_centered_at_zero = homog_center_at_zero(idxs=idxs)
    expected = [[-1, 0, 1], [-1, 0, 1], [0, 0, 0], [1, 1, 1]]
    npt.assert_allclose(idxs_centered_at_zero, expected)


def test_slice():
    volume = torch.rand(3, 3, 3)
    shape = (3, 3)
    slice = slice_atlas(
        shape=shape,
        volume=volume,
        ap=1,
        si=1,
        lr=1,
        rot_frontal=0,
        rot_horizontal=0,
        rot_sagittal=0,
    )
    npt.assert_allclose(slice, volume[1, :, :])


def test_slice_out_of_bounds():
    volume = torch.rand(5, 5, 5)
    shape = (5, 5)
    slice = slice_atlas(
        shape=shape,
        volume=volume,
        ap=2,
        si=1,
        lr=1,
        rot_frontal=0,
        rot_horizontal=0,
        rot_sagittal=0,
    )
    expected = np.zeros(shape)
    expected[1:, 1:] = volume[2, :-1, :-1]
    npt.assert_allclose(slice, expected)


def test_slice_rot_frontal_90():
    volume = torch.rand(3, 3, 3)
    shape = (3, 3)
    slice = slice_atlas(
        shape=shape,
        volume=volume,
        ap=1,
        si=1,
        lr=1,
        rot_frontal=90,
        rot_horizontal=0,
        rot_sagittal=0,
    )
    npt.assert_allclose(slice, np.rot90(volume[1, :, :], k=-1))


# def test_slice_remap_bilinear(atlas):
#     slice = slice_atlas(
#         shape=atlas.shape[1:],
#         volume=torch.tensor(atlas.reference.astype(np.float32)),
#         ap=atlas.reference.shape[0] * 29 / 50,
#         si=atlas.reference.shape[1] / 2,
#         lr=atlas.reference.shape[2] / 2,
#         rot_frontal=0,
#         rot_horizontal=0,
#         rot_sagittal=-6,
#         interpolation="nearest",
#     )
#     Image.fromarray((slice / slice.max() * 255).astype(np.uint8)).save(
#         Path(__file__).resolve().parent / "slice_remap_nearest.jpg"
#     )
#
#     slice = slice_atlas(
#         shape=atlas.shape[1:],
#         volume=torch.tensor(atlas.reference.astype(np.float32)),
#         ap=atlas.reference.shape[0] * 29 / 50,
#         si=atlas.reference.shape[1] / 2,
#         lr=atlas.reference.shape[2] / 2,
#         rot_frontal=0,
#         rot_horizontal=0,
#         rot_sagittal=-6,
#         interpolation="bilinear",
#     )
#     Image.fromarray((slice / slice.max() * 255).astype(np.uint8)).save(
#         Path(__file__).resolve().parent / "slice_remap_bilinear.jpg"
#     )
