"""Imagine fifteen reward steps for each possible first button in a recorded RGB context."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from train_earned_dreamer import digest, write_json
from upstream_emerald import PIN


def imagine_rewards(agent, initial, data):
    import jax
    carry, obs, actions, _ = agent._apply_replay_context(initial, data)
    past = lambda tree: jax.tree.map(lambda array: array[:, :-15], tree)
    history = past(obs)
    _, _, tokens = agent.enc(carry[0], history, history["is_first"], training=False)
    state, _, _ = agent.dyn.observe(carry[1], tokens, past(actions), history["is_first"], training=False)
    future = jax.tree.map(lambda array: array[:, -15:], actions)
    _, features, _ = agent.dyn.imagine(state, future, 15, training=False)
    return initial, dict(estimate=agent.rew(agent.feat2tensor(features), 2).pred())


def predict_buttons(context, past, future, parent, checkpoint, source):
    import elements
    import jax
    from ruamel.yaml import YAML
    n = len(context)
    if (not n or context.dtype != np.uint8 or context.shape != (n, 33, 40, 60, 3)
            or past.shape != (n, 32) or future.shape != (n, 9, 15)
            or not np.issubdtype(past.dtype, np.integer) or not np.issubdtype(future.dtype, np.integer)
            or any(a.min() < 0 or a.max() > 8 for a in (past, future))
            or not np.all(future[:, :, 0] == np.arange(9)) or not np.all(future[:, :, 1:] == future[:, :1, 1:])):
        raise ValueError("Invalid histories or incomparable candidate action sequences")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "dreamerv3", "embodied"], check=True)
    info = json.loads((parent / "provenance.json").read_text())
    if info["upstream_revision"] != PIN or revision != PIN:
        raise ValueError("Expected the pinned official Dreamer revision")
    parent_weights = info["agent_sha256"]
    if digest(parent / "frozen-policy/agent.pkl") != parent_weights:
        raise ValueError("Parent checkpoint differs from its provenance")
    weights = digest(checkpoint / "agent.pkl")
    if weights != parent_weights:
        fit = json.loads((checkpoint.parent / "result.json").read_text())
        if (fit["trained_sha256"], fit["parent_sha256"], fit["upstream_revision"]) != (weights, parent_weights, PIN):
            raise ValueError("Adapted checkpoint differs from its registered parent or fit")
    raw = YAML(typ="safe").load((parent / "config.yaml").read_text())
    if raw["jax"]["platform"] != "cuda" or raw["replay_context"] != 1:
        raise ValueError("Expected CUDA and one step of replay context")
    raw["jax"]["precompile"] = False
    sys.path.insert(0, str(source.resolve()))
    from dreamerv3 import main as upstream
    from dreamerv3.agent import Agent
    from embodied.jax import internal

    class Spaces:
        obs_space = dict(image=elements.Space(np.uint8, (64, 64, 3)), reward=elements.Space(np.float32),
                         is_first=elements.Space(bool), is_last=elements.Space(bool), is_terminal=elements.Space(bool))
        act_space = dict(action=elements.Space(np.int32, (), 0, 9), reset=elements.Space(bool))

        def close(self):
            pass

    with patch.object(upstream, "make_env", lambda *args, **kwargs: Spaces()), patch.object(Agent, "report", imagine_rewards):
        agent = upstream.make_agent(elements.Config(raw))
        devices = [d.device_kind for d in jax.devices() if d.platform == "gpu"]
        if not devices:
            raise RuntimeError("CUDA device unavailable")
        saved = elements.Checkpoint()
        saved.agent = agent
        saved.load(str(checkpoint), keys=["agent"])
        updates = int(agent.n_updates)
        batch = raw["batch_size"]

        def predict(indices, button, seed, future_pixels=0):
            inputs = agent._zeros(agent.spaces, (batch, 48))
            inputs["image"][:, :33, 12:52, 2:62] = context[indices]
            inputs["image"][:, 33:] = future_pixels
            inputs["action"][:, :32] = past[indices]
            inputs["action"][:, 32:47] = future[indices, button]
            inputs["consec"][:] = 1
            assert not inputs["reward"].any()
            data = internal.device_put(inputs, agent.train_sharded)
            data["seed"] = agent._seeds(seed, agent.train_mirrored)
            _, metrics = agent.report(agent.init_report(batch), data)
            return np.asarray(metrics["estimate"])

        forecasts = np.empty((n, 9, 15), np.float32)
        for start in range(0, n, batch):
            indices = np.minimum(np.arange(start, start + batch), n - 1)
            count = min(batch, n - start)
            for button in range(9):
                draws = [predict(indices, button, seed) for seed in range(81703, 81711)]
                if start == 0 and button == 0 and not np.array_equal(draws[0], predict(indices, button, 81703, 255)):
                    raise RuntimeError("An imagined reward depended on future pixels")
                forecasts[start:start + count, button] = np.mean(draws, axis=0)[:count]
        if int(agent.n_updates) != updates or not np.isfinite(forecasts).all():
            raise RuntimeError("Invalid predictions or changed optimizer state")
    return forecasts, dict(weights=weights, gpu=devices, upstream_revision=revision, windows=n,
                           real_reward_input_zero=True, future_image_invariance_verified=True,
                           matched_latent_noise_draws=8, optimizer_updates=0)


def run(training, parent, checkpoint, upstream):
    games = json.loads((training / "completed.json").read_text())["games"]
    arrays = {key: [] for key in ("context", "past", "future")}
    for game in games:
        file = training / game["training_tape"] / "windows.npz"
        if digest(file) != game["data_sha256"]:
            raise ValueError("Training examples changed after collection")
        with np.load(file, allow_pickle=False) as data:
            for key in arrays:
                arrays[key].append(data[key])
    forecasts, info = predict_buttons(*(np.concatenate(v) for v in arrays.values()), parent, checkpoint, upstream)
    temp = training / "world-model-forecasts.tmp.npz"
    np.savez_compressed(temp, forecasts=forecasts)
    destination = training / "world-model-forecasts.npz"
    temp.replace(destination)
    write_json(training / "world-model-forecasts.json", dict(**info, forecasts_sha256=digest(destination),
               training_tapes=[g["training_tape"] for g in games], window_sha256=[g["data_sha256"] for g in games]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("training", "parent", "checkpoint", "upstream"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    run(args.training, args.parent, args.checkpoint, args.upstream)
