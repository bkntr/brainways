# Overview

[![DOI](https://img.shields.io/badge/DOI-10.1101/2023.05.25.542252-green.svg)](https://doi.org/10.1101/2023.05.25.542252)
[![License GNU GPL v3.0](https://img.shields.io/pypi/l/brainways.svg?color=green)](https://github.com/bkntr/brainways/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/brainways.svg?color=green)](https://pypi.org/project/brainways)
[![Python Version](https://img.shields.io/pypi/pyversions/brainways.svg?color=green)](https://python.org)
[![tests](https://github.com/bkntr/brainways/workflows/tests/badge.svg)](https://github.com/bkntr/brainways/actions)
[![codecov](https://codecov.io/gh/bkntr/brainways/branch/main/graph/badge.svg)](https://codecov.io/gh/bkntr/brainways)
[![Documentation Status](https://readthedocs.org/projects/brainways/badge/?version=latest)](https://brainways.readthedocs.io/en/latest/?badge=latest)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/brainways)](https://napari-hub.org/plugins/brainways)

<!-- markdownlint-disable MD026 -->
## What Is Brainways?
<!-- markdownlint-enable MD026 -->

Brainways is an AI-based tool for automated registration, quantification and generation of brain-wide activity networks based on fluorescence in coronal slices.

![Brainways UI](assets/brainways-ui.gif)

<!-- markdownlint-disable MD026 -->
## Why Brainways?
<!-- markdownlint-enable MD026 -->

Coronal slice registration, cell quantification and whole-brain contrast analysis between experimental conditions should be made easily accessible from a single software, without requiring programming experience.
Customization should be made easy by having a highly flexible pythonic backend.

## Getting Started

!!! note
    If you are installing Brainways on Windows and need GPU support, you need to first install PyTorch and TensorBoard with GPU support. Please refer to the [PyTorch](https://pytorch.org/get-started/locally/) and [TensorBoard](https://www.tensorflow.org/install/pip) websites for detailed instructions. After installing these dependencies, you can proceed with the Brainways installation.

To install and run brainways, run the following in your python environment:

```bash
pip install brainways
brainways ui
```

Follow our [getting started guide](02_getting_started.md) for more details.

## How it works

Brainways allows users to register, quantify and provide statistical contrast analysis by following several simple steps:

1. Rigid registration of coronal slices to a 3D atlas.
1. Non-rigid registration of coronal slices to a 3D atlas, to account for individual difference and imperfections in acquisition procedure.
1. Cell detection (using [StarDist](https://github.com/stardist/stardist)).
1. Quantification of cell counts per brain region.
1. Statistical analysis:
    * ANOVA contrast analysis.
    * PLS (Partial Least Square) analysis.
    * Network graph creation.

## Architecture

Brainways is structured as a monorepo with two main components. `brainways` contains all core functionalities. The `brainways.ui` submodule provides the graphical user interface as a [napari](https://napari.org/stable/) plugin. `brainways` can be fully accessed and manipulated through Python code for custom and complex usage scenarios. The code for training, evaluating, and running the automatic registration model is contained within the `brainways-reg-model` package.

## Development Status

Brainways is being actively developed by Ben Kantor of Bartal lab, Tel Aviv University, Israel. Our releases can be found [here](https://github.com/bkntr/brainways/releases).

## Citation

If you use brainways, please cite [Kantor and Bartal (2023)](https://doi.org/10.1101/2023.05.25.542252):

    @article{kantor2023brainways,
      title={Brainways: An Open-Source AI-based Software For Registration and Analysis of Fluorescent   Markers on Coronal Brain Slices},
      author={Kantor, Ben and Ben-Ami Bartal, Inbal},
      journal={bioRxiv},
      pages={2023--05},
      year={2023},
      publisher={Cold Spring Harbor Laboratory}
    }

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"brainways" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/bkntr/brainways/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
