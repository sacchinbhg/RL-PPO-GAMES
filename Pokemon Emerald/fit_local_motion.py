"""Learn which image regions predict the visual effect of controller buttons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from train_earned_dreamer import digest, write_json


def descriptors(images, top=None, left=None):
    if top is not None:
        images = images[:, top:top + 12, left:left + 12]
    n, height, width, _ = images.shape
    pooled = images.reshape(n, height//2, 2, width//2, 2, 3).mean((2, 4), dtype=np.float32)
    return pooled.reshape(n, -1) / 255


def distances(first, second):
    # Normalized squared Euclidean distance in mean-pooled RGB.
    return np.maximum(0., (np.square(first).sum(1)[:, None] + np.square(second).sum(1)
                          - 2 * first @ second.T) / first.shape[1])


def nearest(distance, outcomes):
    if distance.shape[1] < 4:
        raise ValueError("At least four training images are required")
    indices = np.argsort(distance, axis=1, kind="stable")[:, :4]
    local = np.take_along_axis(distance, indices, axis=1)
    if not np.isfinite(local).all():
        raise ValueError("Too few independent training observations")
    weights = 1 / np.maximum(.02, np.sqrt(local))
    return (outcomes[indices] * weights[:, :, None]).sum(1) / weights.sum(1, keepdims=True)


def hashes(images):
    return np.asarray([hashlib.sha256(image).hexdigest() for image in images])


def patches():
    return [(top, left) for top in (0, 4, 8, 12, 16, 20, 24, 28) for left in range(0, 49, 4)]


def fit(images, outcomes):
    if len(images) < 24 or len(images) != len(outcomes):
        raise ValueError("Insufficient motion examples for blocked validation")
    buttons = outcomes.shape[1]
    folds = np.arange(len(images)) * 4 // len(images)
    keys = hashes(images)
    mask = (folds[:, None] == folds[None, :]) | (keys[:, None] == keys[None, :])
    candidates = patches()
    losses = []
    for top, left in candidates:
        features = descriptors(images, top, left)
        distance = distances(features, features)
        distance[mask] = np.inf
        prediction = nearest(distance, outcomes)
        losses.append(np.square(prediction - outcomes).mean(0))
    loss = np.stack(losses)
    best = loss.argmin(0)
    return np.asarray([candidates[i] for i in best]), loss[best, np.arange(buttons)]


def predict(train, outcomes, test, coordinates):
    result = {}
    result["button_frequency"] = np.broadcast_to(outcomes.mean(0), (len(test), outcomes.shape[1]))
    result["whole_screen"] = nearest(distances(descriptors(test), descriptors(train)), outcomes)
    local = np.empty_like(result["whole_screen"])
    for j, (top, left) in enumerate(coordinates):
        distance = distances(descriptors(test, top, left), descriptors(train, top, left))
        local[:, j] = nearest(distance, outcomes[:, [j]])[:, 0]
    result["local_screen"] = local
    return result


class ScrollModel:
    def __init__(self, folder):
        identity = json.loads((folder / "fit.json").read_text())
        path = folder / "model.npz"
        if identity["model_sha256"] != digest(path):
            raise ValueError("Stored image-motion model changed")
        self.sha = identity["model_sha256"]
        with np.load(path, allow_pickle=False) as model:
            self.buttons = model["buttons"]
            self.patches = model["patches"]
            self.outcomes = model["scroll"]
            self.features = [descriptors(model["train"], top, left) for top, left in self.patches]

    def predictions(self, frame):
        result = np.empty(len(self.buttons), np.float32)
        for j, (top, left) in enumerate(self.patches):
            current = descriptors(frame[None], top, left)
            distance = distances(current, self.features[j])
            result[j] = nearest(distance, self.outcomes[:, [j]])[0, 0]
        return result


def load(folder):
    meta = json.loads((folder / "completed.json").read_text())
    path = folder / "examples.npz"
    if digest(path) != meta["data_sha256"]:
        raise ValueError("Motion training file changed")
    with np.load(path, allow_pickle=False) as data:
        images, outcomes, at = data["rgb"], data["scroll"], data["decision"]
        if not np.array_equal(data["button"], meta["buttons"]):
            raise ValueError("Action order changed")
    if (len(images) != meta["positions"] or images.dtype != np.uint8 or images.shape[1:] != (40, 60, 3)
            or outcomes.dtype != bool or outcomes.shape != (len(images), len(meta["buttons"]))
            or not (np.diff(at) > 0).all() or np.any(at <= meta["skip_prefix"])):
        raise ValueError("Invalid recorded motion examples")
    return images, outcomes.astype(np.float32), meta


def run(source, heldout, output):
    train, outcomes, first = load(source)
    output.mkdir(parents=True, exist_ok=True)
    coordinates, losses = fit(train, outcomes)
    temp = output / "model.tmp.npz"
    model = output / "model.npz"
    np.savez_compressed(temp, train=train, scroll=outcomes, patches=coordinates, buttons=np.asarray(first["buttons"]))
    temp.replace(model)
    write_json(output / "fit.json", dict(source=first["data_sha256"], model_sha256=digest(model),
               positions=len(train), patches=coordinates.tolist(), training_validation_error=losses.tolist()))
    test, actual, second = load(heldout)
    if first["training_tape"] == second["training_tape"] or first["buttons"] != second["buttons"] or first["motion_probe"] != second["motion_probe"]:
        raise ValueError("Expected independent games and a common action definition")
    unique = ~np.isin(hashes(test), hashes(train))
    sensitive = unique & (actual.min(1) != actual.max(1))
    if not sensitive.any():
        raise ValueError("No independent action-sensitive test positions")
    reports = {}
    for label, prediction in predict(train, outcomes, test, coordinates).items():
        chosen = np.isclose(prediction, prediction.max(1, keepdims=True), atol=1e-7, rtol=0.)
        realized = (chosen * actual).sum(1) / chosen.sum(1)
        reports[label] = dict(realized_scroll=float(realized[sensitive].mean()),
                              mean_squared_error=float(np.square(prediction[unique] - actual[unique]).mean()))
    uniform = float(actual[sensitive].mean())
    best_baseline = max(uniform, *(reports[k]["realized_scroll"] for k in ("button_frequency", "whole_screen")))
    write_json(output / "heldout.json", dict(test_source=second["data_sha256"], heldout_positions=int(unique.sum()),
               trained_on_test=False, excluded_exact_screens=int((~unique).sum()), action_sensitive_positions=int(sensitive.sum()),
               uniform=uniform, results=reports, model_sha256=digest(model),
               local_improvement_over_strongest_baseline=reports["local_screen"]["realized_scroll"]-best_baseline))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "heldout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.training, args.heldout, args.output)
