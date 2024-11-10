from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtWidgets import QWidget

from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.pipeline.brainways_pipeline import BrainwaysPipeline

if TYPE_CHECKING:
    from brainways.ui.brainways_ui import BrainwaysUI


class Controller(ABC):
    def __init__(self, ui: BrainwaysUI):
        self.ui = ui
        self.widget: Optional[QWidget] = None
        self._is_open = False

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None: ...

    @abstractmethod
    def default_params(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams: ...

    @staticmethod
    @abstractmethod
    def has_current_step_params(params: BrainwaysParams) -> bool: ...

    @staticmethod
    @abstractmethod
    def enabled(params: BrainwaysParams) -> bool: ...

    def pipeline_loaded(self) -> None:
        pass

    @abstractmethod
    def run_model(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams: ...

    @property
    @abstractmethod
    def params(self) -> BrainwaysParams: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def _check_is_open(self):
        if not self._is_open:
            raise RuntimeError("Controller is not open")

    @property
    def pipeline(self) -> BrainwaysPipeline:
        return self.ui.project.pipeline
