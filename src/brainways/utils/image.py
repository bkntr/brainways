from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_opening
from sklearn.cluster import MiniBatchKMeans

Box = Tuple[float, float, float, float]


def normalize_min_max(image: Union[np.ndarray, torch.Tensor]):
    return (image - image.min()) / (image.max() - image.min())


def normalize_contrast(
    image: np.ndarray, min_quantile: float = 1.0, max_quantile: float = 99.8
):
    min_val, max_val = np.percentile(image, [min_quantile, max_quantile])
    image = np.clip(image, min_val, max_val)
    image = normalize_min_max(image)
    return image


def convert_to_uint8(image: np.ndarray, normalize: bool = True):
    if normalize:
        image = normalize_min_max(image)
    image = (image * 255).round().astype(np.uint8)
    return image


ImageSizeHW = Tuple[int, int]


def get_resize_size(
    input_size: ImageSizeHW,
    output_size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    keep_aspect: bool = False,
) -> ImageSizeHW:
    if (output_size is None) == (scale is None):
        raise ValueError("Please set either output_size or scale")
    if output_size is None:
        output_size = (int(input_size[0] * scale), int(input_size[1] * scale))
    if keep_aspect:
        h_factor = input_size[0] / output_size[0]
        w_factor = input_size[1] / output_size[1]
        factor = max(h_factor, w_factor)
        output_size = (
            int(round(input_size[0] / factor)),
            int(round(input_size[1] / factor)),
        )
    return output_size


def resize_image(
    image: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    keep_aspect: bool = False,
):
    output_size = get_resize_size(
        input_size=image.shape,
        output_size=size,
        scale=scale,
        keep_aspect=keep_aspect,
    )
    if tuple(output_size) <= (image.shape[0], image.shape[1]):
        factor = min(image.shape[0] / output_size[0], image.shape[1] / output_size[1])
        for i in range(int(np.log2(factor))):
            down_h, down_w = int(image.shape[0] / 2), int(image.shape[1] / 2)
            factor /= 2
            image = cv2.resize(image, dsize=(down_w, down_h))
    image = cv2.resize(image, dsize=(output_size[1], output_size[0]))
    return image


def brain_mask(image: np.ndarray):
    # quantize image to black and white
    h, w = image.shape[:2]
    image_flat = image.reshape((h * w, 1))
    kmeans = MiniBatchKMeans(n_clusters=2)
    labels = kmeans.fit_predict(image_flat)
    labels_order = kmeans.cluster_centers_.flatten().argsort()
    quantized = labels_order[labels].astype("uint8").reshape((h, w))

    if (quantized == 0).all():
        return np.zeros_like(image, dtype=bool)

    mask = np.ones_like(image, dtype=bool)

    connected_components, nr_objects = ndimage.label(quantized)
    cc_sizes = {
        cc: (connected_components == cc).sum() for cc in range(1, nr_objects + 1)
    }
    largest_cc_size = max(cc_sizes.values())

    # remove small artifacts around the edges
    edges = np.ones_like(image, dtype=bool)
    edges[1:-1, 1:-1] = False
    edge_ccs = set(connected_components[edges]) - {0}
    for cc in edge_ccs:
        if cc_sizes[cc] < largest_cc_size * 0.5:
            mask[connected_components == cc] = False

    # remove speckles artifacts
    for cc, cc_size in cc_sizes.items():
        if cc_size < (image.shape[0] * image.shape[1]) * 0.01:
            mask[connected_components == cc] = False

    # remove background around the edges
    connected_components, nr_objects = ndimage.label(1 - quantized)
    edge_ccs = set(connected_components[edges]) - {0}
    for cc in edge_ccs:
        mask[connected_components == cc] = False

    # remove left artifacts with opening
    mask = binary_opening(mask, np.ones((5, 5)))

    return mask.astype(bool)


def brain_mask_simple(image: np.ndarray):
    # quantize image to black and white
    h, w = image.shape[:2]
    image_flat = image.reshape((h * w, 1))
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=0)  # deterministic
    labels = kmeans.fit_predict(image_flat)
    labels_order = kmeans.cluster_centers_.flatten().argsort()
    quantized = labels_order[labels].astype("uint8").reshape((h, w))

    if (quantized == 0).all():
        return np.zeros_like(image, dtype=bool)

    return quantized.astype(bool)


def nonzero_bounding_box(image: np.ndarray):
    ys, xs = np.nonzero(image)
    if len(ys) == 0:
        return 0, 0, image.shape[1], image.shape[0]
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return x0, y0, x1 - x0, y1 - y0


def nonzero_bounding_box_tensor(image: torch.Tensor):
    idxs = torch.nonzero(image)
    if len(idxs) == 0:
        return 0, 0, image.shape[1], image.shape[0]

    (y0, x0), _ = idxs.min(dim=0)
    (y1, x1), _ = idxs.max(dim=0)
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def crop_nonzero(image: np.ndarray):
    x, y, w, h = nonzero_bounding_box(image)
    return image[y : y + h, x : x + w]


def annotation_outline(annotation: torch.Tensor):
    outline = kornia.filters.spatial_gradient(
        annotation[None, None].float(), mode="diff"
    )
    outline = (outline[0, 0].abs().amax(dim=0) > 0).byte() * 255
    return outline


def slice_contrast_values(slice_image: np.ndarray):
    min_val, max_val = np.percentile(slice_image[slice_image > 0], [1, 98])
    return min_val, max_val


def slice_to_uint8(slice_image: np.ndarray):
    min_val, max_val = slice_contrast_values(slice_image)
    slice_image = np.clip(slice_image, min_val, max_val)
    slice_image = normalize_min_max(slice_image)
    slice_image = convert_to_uint8(slice_image)
    return slice_image


def slice_sharpness(
    image: np.ndarray,
    scales: int = 3,
    gradient_clip_value: Optional[int] = 100,
) -> float:
    mask = brain_mask_simple(image).astype(float)
    scale_sharpnesses = []
    for i in range(scales):
        if i > 0:
            image = resize_image(image, scale=0.5)
            mask = resize_image(mask, scale=0.5).round()
        gy, gx = np.gradient(image)
        if gradient_clip_value is not None:
            gy = np.minimum(gy, gradient_clip_value)
            gx = np.minimum(gx, gradient_clip_value)
        gnorm = np.sqrt(gx**2 + gy**2)
        scale_sharpnesses.append(np.sum(gnorm * mask) / np.sum(mask))
    sharpness = np.mean(scale_sharpnesses).item()
    return sharpness
