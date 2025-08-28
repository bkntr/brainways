# Brainways

[![DOI](https://img.shields.io/badge/DOI-10.1101/2023.05.25.542252-green.svg)](https://doi.org/10.1101/2023.05.25.542252)
[![License GNU GPL v3.0](https://img.shields.io/pypi/l/brainways.svg?color=green)](https://github.com/bkntr/brainways/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/brainways.svg?color=green)](https://pypi.org/project/brainways)
[![Python Version](https://img.shields.io/pypi/pyversions/brainways.svg?color=green)](https://python.org)
[![tests](https://github.com/bkntr/brainways/workflows/tests/badge.svg)](https://github.com/bkntr/brainways/actions)
[![codecov](https://codecov.io/gh/bkntr/brainways/branch/main/graph/badge.svg)](https://codecov.io/gh/bkntr/brainways)
[![Documentation Status](https://readthedocs.org/projects/brainways/badge/?version=latest)](https://brainways.readthedocs.io/en/latest/?badge=latest)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/brainways)](https://napari-hub.org/plugins/brainways)

## Overview

Brainways is an AI-powered tool designed for the automated analysis of brain-wide activity networks from fluorescence imaging in coronal slices. It streamlines the process of registration, cell quantification, and statistical comparison between experimental groups, all accessible through a user-friendly interface without requiring programming expertise. For advanced users, Brainways also offers a flexible Python backend for customization.

![Brainways User Interface Demo](assets/brainways-ui.gif)

## Key Features

Brainways simplifies complex analysis workflows into manageable steps:

1.  **Rigid Registration:** Aligns coronal slices to a 3D reference atlas.
2.  **Non-rigid Registration:** Refines alignment to account for individual variations and tissue distortions.
3.  **Cell Detection:** Automatically identifies cells using the [StarDist](https://github.com/stardist/stardist) algorithm.
4.  **Quantification:** Counts cells within defined brain regions.
5.  **Statistical Analysis:**
    *   Performs ANOVA contrast analysis between experimental conditions.
    *   Conducts Partial Least Squares (PLS) analysis.
    *   Generates network graphs visualizing brain-wide activity patterns.

## Getting Started

!!! note "Windows GPU Support Pre-installation"
    If you plan to use Brainways with GPU acceleration on Windows, you must install GPU-compatible versions of PyTorch and TensorFlow *before* installing Brainways. Follow the instructions on the [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install/pip) websites. Once these dependencies are met, proceed with the Brainways installation below.

Install and launch the Brainways user interface using pip:

```bash
pip install brainways
brainways ui
```

For a detailed walkthrough, please refer to our [Getting Started Guide](https://brainways.readthedocs.io/en/latest/02_getting_started/).

!!! tip "Achieving Reliable Results"
    To ensure the best possible outcomes with Brainways, we highly recommend reviewing our [Best Practices Guide](https://brainways.readthedocs.io/en/latest/04_best_practices/).

## Architecture

Brainways is built as a monorepo containing two primary components:

*   `brainways`: The core library housing all backend functionalities, including registration algorithms, quantification logic, and statistical tools. It can be used programmatically via Python for custom workflows. The automatic registration model inference code resides within the `brainways.model` subpackage.
*   `brainways.ui`: A [napari](https://napari.org/stable/) plugin providing the graphical user interface for interactive analysis.

## Development Status

Brainways is under active development by Ben Kantor at the Bartal Lab, Tel Aviv University, Israel. Check out our [releases page](https://github.com/bkntr/brainways/releases) for the latest updates.

## Citation

If Brainways contributes to your research, please cite our publication: [Kantor and Bartal (2025)](https://doi.org/10.1038/s41386-025-02105-3).

```bibtex
@article{kantor2025mapping,
    title={Mapping brain-wide activity networks: brainways as a tool for neurobiological discovery},
    author={Kantor, Ben and Ruzal, Keren and Ben-Ami Bartal, Inbal},
    journal={Neuropsychopharmacology},
    pages={1--11},
    year={2025},
    publisher={Springer International Publishing Cham}
}
```

## License

Brainways is distributed under the terms of the [GNU GPL v3.0] license. It is free and open-source software.

## Issues and Support

Encountering problems? Please [file an issue] on our GitHub repository with a detailed description of the problem.

[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[file an issue]: https://github.com/bkntr/brainways/issues
