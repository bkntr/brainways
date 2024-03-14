from pathlib import Path

PACKAGE_ROOT = Path(__file__).absolute().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
TRAINED_MODEL_ROOT = PROJECT_ROOT / "trained_model"
SYNTH_TRAINED_MODEL_ROOT = TRAINED_MODEL_ROOT / "synth"
REAL_TRAINED_MODEL_ROOT = TRAINED_MODEL_ROOT / "real"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DATA_ROOT = PROJECT_ROOT / "data"
SYNTH_DATA_ROOT = DATA_ROOT / "synth"
SYNTH_DATA_ZIP_PATH = DATA_ROOT / "synth.zip"
REAL_DATA_ROOT = DATA_ROOT / "real"
REAL_DATA_ZIP_PATH = DATA_ROOT / "real.zip"
REAL_RAW_DATA_ROOT = DATA_ROOT / "real_raw_tif"
