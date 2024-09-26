"""
Code adopted from https://github.com/SuperElastix/elastix_napari
Author: Viktor van der Valk, v.o.van_der_valk@lumc.nl
"""
import tempfile
from pathlib import Path
from typing import Optional

import itk
import numpy as np

_ELASTIX_PARAM_FILES = [
    # Path(__file__).parent / "Par0033similarity.txt",
    # Path(__file__).parent / "Par0033bspline.txt",
    # Path(__file__).parent / "ElastixParameterAffine.txt",
    # Path(__file__).parent / "ElastixParameterBSpline.txt",
    Path(__file__).parent / "align_affine.txt",
    Path(__file__).parent / "align_bspline.txt",
]


def elastix_registration(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_points: np.ndarray,
    fixed_mask: Optional[np.ndarray] = None,
    moving_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Takes user input and calls elastix' registration function in itkelastix.
    """
    if fixed is None or moving is None:
        raise ValueError("No images selected for registration.")

    # Convert image layer to itk_image
    fixed = itk.image_view_from_array(fixed)
    fixed = fixed.astype(itk.F)
    moving = itk.image_view_from_array(moving)
    moving = moving.astype(itk.F)

    if fixed_mask is not None:
        fixed_mask = itk.image_view_from_array(fixed_mask)
        fixed_mask = fixed_mask.astype(itk.UC)
    if moving_mask is not None:
        moving_mask = itk.image_view_from_array(moving_mask)
        moving_mask = moving_mask.astype(itk.UC)

    parameter_object = itk.ParameterObject.New()
    # for param_file in _ELASTIX_PARAM_FILES:
    #     parameter_object.AddParameterFile(str(param_file))

    default_bspline_parameter_map = parameter_object.GetDefaultParameterMap("spline")
    parameter_object.AddParameterMap(default_bspline_parameter_map)

    fixed_points_str = "\n".join(f"{int(x)} {int(y)}" for x, y in fixed_points)
    points_file_contents = f"point\n{len(fixed_points)}\n" + fixed_points_str

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(points_file_contents)
        tmpfilename = f.name

    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed,
        moving,
        parameter_object,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        log_to_console=True,
        fixed_point_set_file_name=tmpfilename,
    )

    Path(tmpfilename).unlink()

    registered_points = np.array(
        [
            float(x)
            for x in result_transform_parameters.GetParameterMap(0).asdict()[
                "TransformParameters"
            ]
        ]
    ).reshape(-1, 2)

    return registered_points
