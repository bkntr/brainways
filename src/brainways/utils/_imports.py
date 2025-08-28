# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General utilities. Adapted from PytorchLightining"""

import importlib
from importlib.util import find_spec


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> _package_available('os')
    True
    >>> _package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def _module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('os.bla')
    False
    >>> _module_available('bla.bla')
    False
    """
    module_names = module_path.split(".")
    if not _package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


NAPARI_AVAILABLE = _module_available("napari")
