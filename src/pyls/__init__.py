from ._version import get_versions
from .io import load_results, save_results
from .matlab import import_matlab_result
from .structures import PLSInputs, PLSResults
from .types import behavioral_pls, meancentered_pls, pls_regression

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "__version__",
    "behavioral_pls",
    "meancentered_pls",
    "pls_regression",
    "import_matlab_result",
    "save_results",
    "load_results",
    "examples",
    "PLSInputs",
    "PLSResults",
]
