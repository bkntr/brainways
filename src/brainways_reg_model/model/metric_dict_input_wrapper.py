from typing import Dict

from torch import Tensor
from torchmetrics import Metric


class MetricDictInputWrapper(Metric):
    def __init__(self, metric: Metric, key: str) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise TypeError(
                "metric arg need to be an instance of a torchmetrics metric"
                f" but got {metric}"
            )
        self._base_metric = metric
        self.key = key

    def update(self, *args: Dict, **kwargs: Dict) -> None:
        new_args = [a[self.key] for a in args]
        new_kwargs = {k: v[self.key] for k, v in kwargs.items()}
        self._base_metric(*new_args, **new_kwargs)

    def compute(self) -> Dict[str, Tensor]:
        return self._base_metric.compute()

    def reset(self) -> None:
        self._base_metric.reset()

    def persistent(self, mode: bool = True) -> None:
        self._base_metric.persistent(mode)
