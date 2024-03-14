import pytest
import torch

from brainways.utils.image import ImageSizeHW, annotation_outline, get_resize_size


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
            [
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
            ],
        ),
        (
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ],
            [
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
                [0, 255, 255, 0, 0],
            ],
        ),
    ],
)
def test_annotation_outline(annotation, expected):
    annotation = torch.as_tensor(annotation)
    outline = annotation_outline(annotation)
    expected = torch.as_tensor(expected).byte()
    torch.testing.assert_close(outline, expected)


@pytest.mark.parametrize(
    "input_size,output_size,scale,keep_aspect,expected",
    [
        ((10, 10), (10, 10), None, False, (10, 10)),
        ((10, 10), (10, 10), None, True, (10, 10)),
        ((10, 10), (10, 5), None, False, (10, 5)),
        ((10, 10), (5, 10), None, False, (5, 10)),
        ((10, 10), (10, 5), None, True, (5, 5)),
        ((10, 10), (25, 20), None, True, (20, 20)),
        ((10, 10), (25, 20), None, False, (25, 20)),
        ((10, 10), (5, 10), None, True, (5, 5)),
        ((10, 10), None, 1.0, False, (10, 10)),
        ((10, 10), None, 1.0, True, (10, 10)),
        ((10, 10), None, 0.5, False, (5, 5)),
        ((10, 10), None, 0.5, True, (5, 5)),
        ((10, 10), None, 2.0, False, (20, 20)),
        ((10, 10), None, 2.0, True, (20, 20)),
    ],
)
def test_get_resize_size(
    input_size: ImageSizeHW,
    output_size: ImageSizeHW,
    scale: float,
    keep_aspect: bool,
    expected: ImageSizeHW,
):
    result = get_resize_size(
        input_size=input_size,
        output_size=output_size,
        scale=scale,
        keep_aspect=keep_aspect,
    )
    assert result == expected
