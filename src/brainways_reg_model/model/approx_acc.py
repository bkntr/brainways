import torch
from torchmetrics import Metric


class ApproxAccuracy(Metric):
    def __init__(self, tolerance: float = 10.0):
        super().__init__()

        self.tolerance = tolerance
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.correct += torch.sum(torch.abs(preds - target) <= self.tolerance)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
