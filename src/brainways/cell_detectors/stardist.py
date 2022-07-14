from csbdeep.utils import normalize
from stardist.models import StarDist2D


class StarDist:
    def __init__(self):
        self.model = StarDist2D.from_pretrained("2D_versatile_fluo")

    def run(self, image):
        img = normalize(image, 1, 99.8)
        labels, details = self.model.predict_instances(img)
        return labels
