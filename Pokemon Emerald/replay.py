"""Replay one archived button trajectory from boot to validate its endpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from emerald import Emerald
from experiment import gallery, location, progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--map", help="Choose the shortest stored path to this map")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, help="Save example frames observed along this route")
    options = parser.parse_args()
    if options.out.exists() or (options.gallery and options.gallery.exists()):
        parser.error("Choose a new output filename")
    paths = json.loads(Path(__file__).with_name("local.json").read_text(encoding="utf-8-sig"))
    data = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    if hashlib.sha256(Path(paths["core"]).read_bytes()).hexdigest() != data["spec"]["core_sha256"]:
        parser.error("mGBA core build differs from the training run")
    candidates = [(key, cell) for key, cell in data["archive"].items()
                  if not options.map or location(key) == options.map]
    if not candidates:
        parser.error("Archive contains no matching cells")
    target, cell = min(candidates, key=lambda entry: (
        0 if options.map else -progress(entry[0]), len(entry[1]["path"])))
    env = Emerald(paths["core"], paths["rom"])
    try:
        key = env.descriptor()
        frames = {}
        for action in cell["path"]:
            image, key = env.step(action)
            if options.gallery and (key not in frames or image.std() > frames[key]["image"].std()):
                frames[key] = {"image": image}
        report = dict(start="fresh boot", input_source="archived controller sequence",
                      decisions=len(cell["path"]), target=target, actual=key,
                      target_map=location(target), actual_map=location(key), matched=target == key)
        options.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if options.gallery:
            gallery(frames, options.gallery)
        print(json.dumps(report))
    finally:
        env.close()


if __name__ == "__main__":
    main()
