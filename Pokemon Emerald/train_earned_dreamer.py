"""Train the pinned DreamerV3 world model, actor and critic on verified self-earned episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from upstream_emerald import PIN


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class Trajectory:
    def __init__(self, folder):
        self.folder = folder
        self.source = json.loads((folder / "manifest.json").read_text())
        audit, = (folder / "audit").glob("*.json")
        self.audit = json.loads(audit.read_text())
        self.tape = audit.with_suffix(".bin").read_bytes()
        if (self.source["frozen_evaluation_data_used"] or self.source["replay_restores_during_game"] != 0
                or hashlib.sha256(self.tape).hexdigest() != self.source["source_sha256"]
                or not self.tape or len(self.tape) != self.source["decisions"] or len(self.tape) != self.audit["steps"]):
            raise ValueError(f"Invalid verified training trajectory: {folder}")
        official=(self.source.get("source_kind")=="official_training_worker"
                  and self.source.get("stored_replay_actions_verified") is True
                  and self.source.get("cold_replay_rewards_verified")==len(self.tape))
        if not official and (not self.audit["uncapped_exploration"] or self.source["source_training_forks"]<1):
            raise ValueError("Expected a verified self-generated training run")

    def episode(self, uncapped=False):
        """Emit (image, outgoing action, incoming reward, first, last, terminal) in driver order."""
        if uncapped and not self.audit["uncapped_exploration"]:
            raise ValueError("Capped training logs do not contain later first-visit rewards")
        cursor = raw_return = visits = 0
        incoming, done, previous = 0., False, None
        for path in sorted(self.folder.glob("chunk-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                image, action, reward, end = (data[k] for k in ("image", "action", "reward", "terminal"))
            n = len(action)
            if (not n or int(path.stem.split("-")[1]) != cursor or action.dtype != np.uint8
                    or action.tobytes() != self.tape[cursor:cursor+n]
                    or image.dtype != np.uint8 or image.shape != (n+1, 40, 60, 3)
                    or reward.shape != (n,) or end.shape != (n,)
                    or previous is not None and not np.array_equal(previous, image[0])):
                raise ValueError(f"Discontinuous or invalid training chunk: {path}")
            for i in range(n):
                if done:
                    raise ValueError("The recorded game continued after a terminal transition")
                yield image[i], int(action[i]), incoming, cursor == 0, False, False
                cents = round(100 * float(reward[i]))
                if cents < 0 or cents % 25 not in (0, 2):
                    raise ValueError(f"Unexpected original Emerald scalar reward: {reward[i]}")
                raw_return += cents
                if cents % 25 == 2:
                    visits += 1
                    if not uncapped and visits > 50:
                        cents -= 2
                incoming, done = cents / 100, bool(end[i])
                cursor += 1
            previous = image[-1].copy()
        if (cursor != len(self.tape) or raw_return != self.source["return_cents"]
                or visits != self.audit["first_visit_cells"] or done != ("stone_badge" in self.audit["events"])):
            raise ValueError("Final training episode disagrees with its original-ROM audit")
        yield previous, 0, incoming, False, True, done

    def repeat(self, uncapped):
        while True:
            yield from self.episode(uncapped)


def populate(batch, lanes, last_actions):
    # The upstream recurrent stream consumes one context position and keeps the real carry.
    # Its first retained image at t is paired with the action that produced it at t-1.
    batch["consec"].fill(1)
    batch["action"][:, 0] = last_actions
    for lane, rows in enumerate(lanes):
        for t in range(1, batch["action"].shape[1]):
            image, button, incoming, first, last, terminal = next(rows)
            batch["image"][lane, t, 12:52, 2:62] = image
            batch["action"][lane, t] = button
            batch["reward"][lane, t] = incoming
            batch["is_first"][lane, t] = first
            batch["is_last"][lane, t] = last
            batch["is_terminal"][lane, t] = terminal
    last_actions[:] = batch["action"][:, -1]


def write_json(path, value):
    draft = path.with_suffix(".tmp")
    draft.write_text(json.dumps(value, indent=2, allow_nan=False))
    draft.replace(path)


def train(datasets, parent, output, upstream_source, updates):
    import elements
    import jax
    from ruamel.yaml import YAML
    if updates < 2:
        raise ValueError("Use at least two optimizer updates to collect loss metrics")
    source_sha = subprocess.check_output(["git", "-C", str(upstream_source), "rev-parse", "HEAD"], text=True).strip()
    if source_sha != PIN:
        raise ValueError("Unexpected upstream Dreamer revision")
    subprocess.run(["git", "-C", str(upstream_source), "diff", "--exit-code", "HEAD", "--", "dreamerv3", "embodied"], check=True)
    sys.path.insert(0, str(upstream_source))
    from dreamerv3 import main as upstream
    from embodied.jax import internal
    games = [Trajectory(path) for path in datasets]
    for game in games:
        for _ in game.episode():
            pass
    provenance = json.loads((parent / "provenance.json").read_text())
    weights = parent / "frozen-policy/agent.pkl"
    if provenance["agent_sha256"] != digest(weights):
        raise ValueError("Parent checkpoint checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(upstream_revision=source_sha, parent_sha256=provenance["agent_sha256"],
                parent_training_steps=provenance["training_step"], additional_emulator_steps=0,
                training_tapes=[t.source["source_sha256"] for t in games], training_lengths=[len(t.tape) for t in games],
                uncapped_exploration=provenance.get("uncapped_exploration",False), optimizer_updates=updates,
                sampling="one chronological lane per earned training episode, starting at boot and cycling at its end")
    if len(games) > 1:
        common = next((i for i, choices in enumerate(zip(*(g.tape for g in games)))
                       if len(set(choices)) != 1), min(len(g.tape) for g in games))
        plan["shared_initial_actions"] = common
    setup = output / "experiment.json"
    if setup.exists() and json.loads(setup.read_text()) != plan:
        raise ValueError("Output directory belongs to a different experiment")
    write_json(setup, plan)
    marker = output / "result.json"
    if marker.exists():
        if json.loads(marker.read_text())["trained_sha256"] != digest(output / "frozen-policy/agent.pkl"):
            raise ValueError("Saved adapted checkpoint checksum mismatch")
        return
    raw = YAML(typ="safe").load((parent / "config.yaml").read_text())
    if raw["jax"]["platform"] != "cuda" or raw["replay_context"] != 1:
        raise ValueError("Expected the CUDA configuration with one replay context step")
    raw["logdir"] = str(output)
    raw["jax"].update(precompile=False, enable_policy=False)
    raw["batch_size"] = len(games)
    length = raw["batch_length"]
    class Spaces:
        obs_space = dict(image=elements.Space(np.uint8, (64, 64, 3)), reward=elements.Space(np.float32),
                         is_first=elements.Space(bool), is_last=elements.Space(bool), is_terminal=elements.Space(bool))
        act_space = dict(action=elements.Space(np.int32, (), 0, 9))
        def close(self):
            pass
    with patch.object(upstream, "make_env", lambda *args, **kwargs: Spaces()):
        agent = upstream.make_agent(elements.Config(raw))
    gpu = [d.device_kind for d in jax.devices() if d.platform == "gpu"]
    if not gpu:
        raise RuntimeError("The Dreamer update requires the WSL CUDA GPU")
    with weights.open("rb") as file:
        original = pickle.load(file)
    agent.load(original)
    start = int(agent.n_updates)
    lanes = [t.repeat(plan["uncapped_exploration"]) for t in games]
    previous = np.zeros(len(lanes), np.int32)
    carry = agent.init_train(len(lanes))
    last_metrics = {}
    for step in range(1, updates+1):
        data = agent._zeros(agent.spaces, (len(lanes), length+1))
        populate(data, lanes, previous)
        device = internal.device_put(data, agent.train_sharded)
        seed = int(agent.n_batches)
        agent.n_batches.increment()
        device["seed"] = agent._seeds(seed, agent.train_mirrored)
        carry, _, metrics = agent.train(carry, device)
        if metrics:
            last_metrics = {k:float(v) for k,v in metrics.items() if k.startswith("loss/") and np.ndim(v) == 0}
            if not last_metrics or not all(np.isfinite(list(last_metrics.values()))):
                raise RuntimeError(f"Missing or non-finite official losses at update {step}")
        if step == 1 or step % 32 == 0 or step == updates:
            write_json(output / "status.json", dict(optimizer_updates=step,
                observations_trained=step*len(lanes)*length, last_completed_losses=last_metrics))
    new = agent.save()
    if (int(agent.n_updates) != start + updates or
            new["counters"]["actions"] != original["counters"]["actions"] or not last_metrics):
        raise RuntimeError("Unexpected learner counters")
    changed = sorted({k.split("/")[0] for k in original["params"]
                      if not np.array_equal(original["params"][k], new["params"][k])})
    if not {"dyn", "enc", "rew", "pol", "val"} <= set(changed):
        raise RuntimeError(f"Missing world-model, actor or critic parameter changes: {changed}")
    checkpoint = output / "frozen-policy"
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp) / "checkpoint"
        stage.mkdir()
        with (stage / "agent.pkl").open("wb") as file:
            pickle.dump(new, file, protocol=4)
        shutil.copyfile(parent / "frozen-policy/step.pkl", stage / "step.pkl")
        (stage / "done").touch()
        if checkpoint.exists():
            if digest(checkpoint / "agent.pkl") != digest(stage / "agent.pkl"):
                raise ValueError("An existing checkpoint differs from this run")
        else:
            stage.rename(checkpoint)
    write_json(marker, dict(**plan, trained_sha256=digest(checkpoint / "agent.pkl"), gpu=gpu,
        imagination_length=raw["agent"]["imag_length"], batch_length=length,
        observations_trained=len(lanes)*length*updates, changed_module_groups=changed,
        last_completed_losses=last_metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    args = parser.parse_args()
    train([p.resolve() for p in args.dataset], args.source_run.resolve(), args.output.resolve(),
          args.upstream.resolve(), args.updates)
