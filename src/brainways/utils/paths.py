from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
ANNOTATE_V1_1_ROOT = DATA_ROOT / "annotate_v1.1"
REG_MODEL = PROJECT_ROOT / "outputs/reg/real/model.ckpt"
