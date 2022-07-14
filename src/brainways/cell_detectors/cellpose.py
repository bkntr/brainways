import logging

from cellpose import models
from cellpose.models import models_logger


class Cellpose:
    def __init__(self):
        models_logger.setLevel(logging.ERROR)
        self.model = models.Cellpose(gpu=True, model_type="nuclei")

    def run(self, image):
        masks, flows, styles, diams = self.model.eval(
            image.squeeze(),
            diameter=20,
            channels=[0, 0],
            net_avg=True,
            flow_threshold=1.0,
            cellprob_threshold=-2,
        )
        return masks
