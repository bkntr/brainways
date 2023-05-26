from __future__ import annotations

import multiprocessing
from typing import Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
from scipy import ndimage
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


def convert_to_uint8(image: Union[np.ndarray, torch.Tensor], normalize: bool = True):
    if normalize:
        image = normalize_min_max(image)
    image = (image * 255).round()
    if isinstance(image, torch.Tensor):
        image = image.to(dtype=torch.uint8)
    else:
        image = image.astype(np.uint8)
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
    min_pixel_value, max_pixel_value = np.quantile(image, (0, 0.85))
    image_flat = np.clip(image_flat, min_pixel_value, max_pixel_value)
    kmeans = MiniBatchKMeans(n_init="auto", n_clusters=2, batch_size=2048)
    labels = kmeans.fit_predict(image_flat)
    labels_order = kmeans.cluster_centers_.flatten().argsort()
    quantized = labels_order[labels].astype("uint8").reshape((h, w))

    if (quantized == 0).all():
        return np.zeros_like(image, dtype=bool)

    # find connected components of quantized image
    connected_components, nr_objects = ndimage.label(quantized)
    cc_indices, cc_sizes = np.unique(connected_components, return_counts=True)

    # remove speckle artifacts
    cc_size_mask = cc_sizes[connected_components]
    no_speckle_mask = cc_size_mask >= (image.shape[0] * image.shape[1]) * 0.01
    quantized *= no_speckle_mask

    # find CCs again after speckle removal
    connected_components, nr_objects = ndimage.label(quantized)
    cc_indices, cc_sizes = np.unique(connected_components, return_counts=True)

    # find largest connected component
    largest_cc_size = max(cc_sizes[1:])

    # remove small artifacts around the edges
    edges_mask = np.ones_like(image, dtype=bool)
    edges_mask[1:-1, 1:-1] = False
    edge_ccs = set(connected_components[edges_mask]) - {0}
    for cc in edge_ccs:
        if cc_sizes[cc] < largest_cc_size * 0.5:
            quantized[connected_components == cc] = False

    return quantized.astype(bool)


def brain_mask_simple(image: np.ndarray):
    # quantize image to black and white
    h, w = image.shape[:2]
    image_flat = image.reshape((h * w, 1))
    kmeans = MiniBatchKMeans(
        n_init="auto",
        n_clusters=2,
        random_state=0,
        batch_size=256 * multiprocessing.cpu_count(),
    )  # deterministic
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


def annotation_outline(annotation: Union[torch.Tensor, np.ndarray]):
    is_ndarray = isinstance(annotation, np.ndarray)
    if is_ndarray:
        annotation = torch.as_tensor(annotation)
    outline = kornia.filters.spatial_gradient(
        annotation[None, None].float(), mode="diff"
    )
    outline = outline.abs().amax(dim=2) > 0
    kernel_size = int(max(annotation.shape) / 512)
    if kernel_size > 1:
        kernel = torch.ones(kernel_size, kernel_size)
        outline = kornia.morphology.dilation(outline, kernel, border_type="constant")
    outline = outline.byte() * 255
    if is_ndarray:
        outline = outline.numpy()
    return outline[0, 0]


def slice_contrast_values(
    slice_image: Union[np.ndarray, torch.Tensor], saturation: float = 0.001
):
    if isinstance(slice_image, torch.Tensor):
        hist, bin_edges = torch.histogram(slice_image.flat, bins=1024)
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
    # min_val, max_val = np.percentile(slice_image[slice_image > 0], [0.001, 99.99])
    return min_display, max_display


def slice_to_uint8(
    slice_image: Union[np.ndarray, torch.Tensor], saturation: float = 0.001
):
    min_val, max_val = slice_contrast_values(slice_image, saturation=saturation)
    if isinstance(slice_image, torch.Tensor):
        slice_image = torch.clip(slice_image, min_val, max_val)
    else:
        slice_image = np.clip(slice_image, min_val, max_val)
    slice_image = convert_to_uint8(slice_image)
    return slice_image


def normalize_contrast_qupath(slice_image: np.ndarray):
    min_val, max_val = slice_contrast_values(slice_image)
    if isinstance(slice_image, torch.Tensor):
        slice_image = torch.clip(slice_image, min_val, max_val)
    else:
        slice_image = np.clip(slice_image, min_val, max_val)
    slice_image = normalize_min_max(slice_image)
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
