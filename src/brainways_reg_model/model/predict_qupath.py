import argparse
import random
from pathlib import Path

import xmltodict
from PIL import Image

from brainways_reg_model.models.reg.model import BrainwaysRegModel


def predict_image(path: Path, model: BrainwaysRegModel):
    image = Image.open(path)
    output = model.predict(image)
    return output


def results_to_xml(model, results):
    results = sorted(results, key=lambda x: -x["outputs"]["ap"])
    xml = {
        "series": {
            "@aligner": "DURACELL v0.1",
            "@first": str(1),
            "@last": str(len(results) + 1),
            "@name": "Section_s001.jpg",
            "slice": [],
        }
    }
    for i, result in enumerate(results):
        ox = model.atlas.shape[2] * 2
        oy = (model.atlas.shape[0] - result["outputs"]["ap"]) * 2
        oz = model.atlas.shape[1] * 2
        ux = -ox
        uy = 0
        uz = 0
        vx = 0
        vy = 0
        vz = -oz
        anchoring = (
            f"ox={ox}&oy={oy}&oz={oz}&ux={ux}&uy={uy}&uz={uz}&vx={vx}&vy={vy}&vz={vz}"
        )
        slice = {
            "@anchoring": anchoring,
            "@filename": result["path"].name,
            "@height": "-999",
            "@nr": str(i + 1),
            "@width": "-999",
        }

        xml["series"]["slice"].append(slice)

    return xmltodict.unparse(xml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--checkpoint", default="trained")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    if args.checkpoint == "trained":
        checkpoint = "outputs/reg/model.ckpt"
    elif args.checkpoint == "training":
        checkpoint = next(
            Path("outputs/reg/lightning_logs/version_0/checkpoints/").glob("*.ckpt")
        )
    else:
        checkpoint = args.checkpoint

    # Load model
    model = BrainwaysRegModel.load_from_checkpoint(checkpoint)
    model.eval()

    # Load paths
    root = Path(args.input)
    if root.is_file():
        paths = [root]
    else:
        paths = list(root.glob("*.jpg"))

    if args.shuffle:
        random.shuffle(paths)

    # Predict
    results = []
    for path in paths:
        results.append({"path": path, "outputs": predict_image(path=path, model=model)})

    xml = results_to_xml(model, results)
    with open(args.output, "w") as f:
        f.write(xml)


if __name__ == "__main__":
    main()
