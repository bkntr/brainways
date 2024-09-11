from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class ImagePath:
    filename: str
    scene: Optional[int] = None
    channel: Optional[int] = None

    def with_channel(self, channel: int):
        return replace(self, channel=channel)

    def __str__(self):
        suffixes = []
        if self.scene is not None:
            suffixes.append(f"Scene #{self.scene}")
        if self.channel is not None:
            suffixes.append(f"Channel #{self.channel}")

        if suffixes:
            suffix = " ".join(suffixes)
            return f"{self.filename} [{suffix}]"
        else:
            return self.filename
