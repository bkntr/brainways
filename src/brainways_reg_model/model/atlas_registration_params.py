from dataclasses import dataclass


@dataclass(frozen=True)
class AtlasRegistrationParams:
    ap: float = 0.0
    rot_frontal: float = 0.0
    rot_horizontal: float = 0.0
    rot_sagittal: float = 0.0
    hemisphere: str = "both"
    confidence: float = 1.0
