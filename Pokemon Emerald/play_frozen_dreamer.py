"""Run the pinned Dreamer policy directly on the WSL GPU and verify its recorded prefixes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

project = Path(__file__).resolve().parent
from pixel_env import PixelPool


def write_json(path, record):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def load_actor(checkpoint, source_run, seed, source, batch_size=4, with_forks=False):
    sys.path.insert(0, str(source))
    import elements
    import jax
    import ninjax as nj
    from ruamel.yaml import YAML
    from dreamerv3.agent import Agent
    from embodied.jax import internal
    from upstream_emerald import PIN
    assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() == PIN
    subprocess.run(["git","-C",str(source),"diff","--exit-code","HEAD","--","dreamerv3","embodied"], check=True)
    config = YAML(typ="safe").load((source_run / "config.yaml").read_text())
    internal.setup(platform="cuda", compute_dtype=config["jax"]["compute_dtype"], prealloc=False, transfer_guard=False)
    assert any(d.platform == "gpu" for d in jax.devices())
    obs = dict(image=elements.Space(np.uint8,(64,64,3)), reward=elements.Space(np.float32),
               is_first=elements.Space(bool), is_last=elements.Space(bool), is_terminal=elements.Space(bool))
    model = object.__new__(Agent)
    agent_config = elements.Config(**config["agent"], replay_context=config["replay_context"],
                                   seed=seed, batch_size=config["batch_size"], batch_length=config["batch_length"])
    model.__init__(obs, dict(action=elements.Space(np.int32, (), 0, 9)), agent_config)
    with (checkpoint / "agent.pkl").open("rb") as file:
        saved = pickle.load(file)
    count = int(saved["counters"]["actions"])
    params = {k:jax.device_put(v) for k,v in saved["params"].items() if k.startswith(("enc/","dyn/","dec/","pol/"))}
    def key(n):
        rng = np.random.default_rng([seed, n])
        return jax.device_put(rng.integers(0, np.iinfo(np.uint32).max, 2, np.uint32))
    _, carry = nj.pure(model.init_policy)(params, batch_size, seed=key(0))
    policy = nj.pure(model.policy)
    @jax.jit
    def apply(params, carry, obs, seed):
        _, (carry, acts, _) = policy(params, carry, obs, "eval", seed=seed)
        return carry, acts["action"]
    def decide(images, reward, first, previous_buttons=None):
        nonlocal carry, count
        if previous_buttons is not None:
            executed = np.asarray(previous_buttons)
            if executed.shape != (batch_size,) or not np.issubdtype(executed.dtype, np.integer) or np.any((executed < 0) | (executed >= 9)):
                raise ValueError("Invalid previously executed actions")
            # The upstream carry stores the action sampled on the last call. A second
            # controller can override that action before it reaches the emulator.
            carry = (*carry[:-1], {"action": jax.device_put(executed.astype(np.int32))})
        padded = np.zeros((batch_size,64,64,3), np.uint8)
        padded[:,12:52,2:62] = images
        obs = dict(image=padded, reward=reward, is_first=np.full(batch_size, first), is_last=np.zeros(batch_size,bool),
                   is_terminal=np.zeros(batch_size,bool))
        carry, acts = apply(params, carry, jax.device_put(obs), key(count))
        count += 1
        return jax.device_get(acts)
    if with_forks:
        def snapshot():
            return carry
        def restore(saved):
            nonlocal carry
            carry=saved
        return decide,snapshot,restore
    return decide


def play(checkpoint, source_run, directory, limit, reference, source, eval_seed=None, log_whiteouts=False):
    directory.mkdir(parents=True, exist_ok=True)
    if limit is not None and limit <= 0:
        raise ValueError("A game must include at least one button decision")
    if limit is None and reference:
        raise ValueError("Use a bounded run to test a historical reference")
    provenance = json.loads((source_run / "provenance.json").read_text())
    with (checkpoint / "agent.pkl").open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    seed = provenance["seed"] if eval_seed is None else eval_seed
    if seed < 0:
        raise ValueError("Evaluation seed must be nonnegative")
    identity = dict(checkpoint=str(checkpoint), weight_sha256=sha, seed=seed,
                    buttons_per_game=limit, games=4, in_game_restores=0, weight_updates=0, platform="WSL CUDA",
                    reference_constraint=str(reference) if reference else None)
    if log_whiteouts:
        identity["external_whiteout_log"]=True
    if provenance.get("battle_feedback",False):
        identity["battle_feedback"]=True
    previous = directory / "provenance.json"
    if previous.exists() and json.loads(previous.read_text()) != identity:
        raise ValueError("Output directory was used for a different policy")
    if (directory / "result.json").is_file():
        return
    if limit is None:
        # A restarted process begins a new game. Preserve earlier partial episodes separately.
        with (directory / "fresh-attempt.json").open("x") as marker:
            json.dump(dict(started=time.time(), weight_sha256=sha),marker)
    write_json(previous, identity)
    ref = None
    if reference:
        meta = json.loads((reference / "provenance.json").read_text())
        assert meta["agent_sha256"] == sha and meta["seed"] == seed
        files = {p.parent.name:p for p in (reference / "external-audit").glob("*/*/*episode-00001.bin")}
        ref = [files[f"env-{i}"].read_bytes() for i in range(4)]
        assert len({len(t) for t in ref}) == 1
    policy = load_actor(checkpoint, source_run, seed, source)
    pool = PixelPool(4, provenance.get("uncapped_exploration", False),directory/"first-maps",log_whiteouts,
                     battle_feedback=provenance.get("battle_feedback",False))
    try:
        images = np.stack([pool.command(i,"reset") for i in range(4)])
        rewards = np.zeros(4, np.float32)
        start = time.monotonic()
        verified_prefix = 0
        steps = itertools.count(1) if limit is None else range(1,limit+1)
        for step in steps:
            buttons = policy(images, rewards, step == 1)
            # Historical buttons are read only AFTER the actor has chosen. A mismatch stops the test.
            if ref and step <= len(ref[0]):
                different = [i for i in range(4) if int(buttons[i]) != ref[i][step-1]]
                if different:
                    failure = dict(decision=step, matched_through=step-1, workers=different,
                                   recorded=[t[step-1] for t in ref], selected=list(map(int, buttons)))
                    for i in different:
                        pool.command(i, "audit", str(directory/"mismatch"/f"env-{i}"))
                    write_json(directory/"prefix-mismatch.json", failure)
                    raise ValueError(f"Reference mismatch at decision {step} on workers {different}")
                verified_prefix = step
            images, rewards, done = pool.step(buttons)
            report = step % 8192 == 0
            full = report and (limit is not None or (step & (step-1)) == 0)
            event_reward = 4 if limit is not None else 1
            if report and limit is None:
                for i in range(4):
                    pool.command(i, "audit_summary", str(directory/"latest"/f"env-{i}"))
            if step == limit or done.any() or full or (rewards >= event_reward).any():
                for i in range(4):
                    if step == limit or done.any() or full or rewards[i] >= event_reward:
                        dest = directory / "games" / f"env-{i}" / f"step-{step:07}"
                        pool.command(i, "audit", str(dest))
                        if done[i]:
                            record, = dest.glob("*.json")
                            proof = dest / "verified-badge"
                            subprocess.run([sys.executable, str(project/"pixel_replay.py"), str(record), str(proof),
                                            "--event","stone_badge","--video"], check=True)
                            badge = json.loads((proof/"verification.json").read_text())
                            assert badge["stone_badge"] == step and badge["roxanne_battle"] < step
                            write_json(directory/"verified-stone-badge.json", {**identity, **badge})
            if step == 256 or step % 1024 == 0 or step == limit or done.any():
                status = dict(decisions_per_game=step, verified_prefix_per_game=verified_prefix,
                              aggregate_buttons=4*step, aggregate_buttons_per_second=4*step/(time.monotonic()-start))
                write_json(directory / "status.json", status)
            if done.any():
                break
        write_json(directory / "result.json", {**identity, **status})
    finally:
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output", type=Path)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--limit", type=int, help="End a comparison game at this many decisions")
    duration.add_argument("--until-badge", action="store_true",
                          help="Continue each fresh game across reporting checkpoints until a verified victory")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seed", type=int, help="Use an independent policy sampling seed for this evaluation")
    parser.add_argument("--log-whiteouts",action="store_true",help="Record game-reported blackouts in the external game audit")
    args = parser.parse_args()
    play(args.checkpoint.resolve(), args.source_run.resolve(), args.output.resolve(), args.limit,
         args.reference.resolve() if args.reference else None, args.upstream.resolve(), args.seed, args.log_whiteouts)
