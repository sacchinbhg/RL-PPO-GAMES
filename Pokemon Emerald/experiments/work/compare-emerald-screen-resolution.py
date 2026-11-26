"""Paired four-seed control for the effect of pixel-state fragmentation on exploration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

work = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("explore", work / "emerald-screen-exploration.py")
explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)


def coarse_key(rgb):
    blocks = rgb.reshape(4, 10, 6, 10, 3).sum(axis=(1, 3), dtype=np.uint16)
    quantized = (blocks // (100 * 64)).astype(np.uint8)
    return hashlib.blake2b(quantized.tobytes(), digest_size=12).digest()


def run():
    folder = explore.out / "coarser-code-control"
    folder.mkdir(exist_ok=True)
    settings = dict(games=list(range(4)), seed=271340, budget_each=32768,
                    change="10x10 pixel blocks and four channel bins, compared with 5x5 and eight",
                    primary="paired story milestones and scalar return at fixed budget",
                    secondary="fraction of buttons chosen after all screen-button pairs were tried",
                    held_out_from_official_training=True)
    manifest = folder / "settings.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == settings
    else:
        manifest.write_text(json.dumps(settings, indent=2))
    original = json.loads((explore.out / "results.json").read_text())["games"]
    rows = []
    for index in settings["games"]:
        alt = explore.play(index, directory=folder / f"game-{index:02}", encode=coarse_key)
        base = next(g for g in original if g["game"] == index)
        assert base["steps"] == alt["steps"] == settings["budget_each"]
        pair = {}
        for name, game in [("original", base), ("coarser", alt)]:
            pair[name] = dict(events=game["events"], scalar_return=game["return_sum"],
                distinct_screens=game["distinct_screen_codes"],
                choices_using_learned_scores=game["steps"] - game["observed_screen_button_pairs"])
        rows.append(dict(game=index, **pair))
        (folder / "paired-progress.json").write_text(json.dumps(rows, indent=2))
    (folder / "comparison.json").write_text(json.dumps(dict(settings=settings, games=rows), indent=2))


if __name__ == "__main__":
    run()
