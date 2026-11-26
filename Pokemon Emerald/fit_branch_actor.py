"""Fit the last official actor layer on a verified self-earned training tape in WSL CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from ruamel.yaml import YAML

from upstream_emerald import PIN

kernel = "pol/head/action/logits/kernel"
bias = "pol/head/action/logits/bias"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def features(params, dataset, config):
    from dreamerv3.agent import Agent
    obs_space = dict(image=elements.Space(np.uint8, (64,64,3)), reward=elements.Space(np.float32),
                     is_first=elements.Space(bool), is_last=elements.Space(bool), is_terminal=elements.Space(bool))
    # Use only the official inference modules, without constructing a second optimizer or replay.
    model = object.__new__(Agent)
    model.__init__(obs_space, dict(action=elements.Space(np.int32, (), 0, 9)), elements.Config(config["agent"]))
    keep = {k: jax.device_put(v) for k, v in params.items() if k.startswith(("enc/", "dyn/", "pol/"))}
    start = nj.pure(lambda: (model.enc.initial(1), model.dyn.initial(1)))
    _, carry = start(keep, seed=0)
    def process(carry, image, first, prevact):
        enc, dyn = carry
        enc, _, tokens = model.enc(enc, {"image": image}, first, training=False)
        dyn, _, feat = model.dyn.observe(dyn, tokens, {"action": prevact}, first, training=False)
        inputs = model.feat2tensor(feat)
        return (enc, dyn), (model.pol.mlp(inputs), model.pol(inputs, 2)["action"].logits)
    pure = nj.pure(process)
    @jax.jit
    def forward(state, seed, carry, image, first, actions):
        return pure(state, carry, image, first, actions, seed=seed)[1]
    hiddens, logits, actions = [], [], []
    previous, offset = 0, 0
    for path in sorted(dataset.glob("chunk-*.npz")):
        with np.load(path) as chunk:
            frames, buttons = chunk["image"], chunk["action"]
            assert len(frames) == len(buttons) + 1
            for i in range(0,len(buttons),64):
                group = buttons[i:i+64]
                n = len(group)
                image = np.zeros((1,64,64,64,3), np.uint8)
                image[0,:n,12:52,2:62] = frames[i:i+n]
                first = np.zeros((1,64), bool)
                first[0,0] = offset == 0
                prev = np.zeros((1,64), np.int32)
                prev[0,0], prev[0,1:n] = previous, group[:-1]
                previous = int(group[-1])
                carry, (hidden, raw) = forward(keep, jax.random.PRNGKey(271800+offset), carry,
                    jax.device_put(image), jax.device_put(first), jax.device_put(prev))
                hiddens.append(np.asarray(jax.device_get(hidden[0,:n])))
                logits.append(np.asarray(jax.device_get(raw[0,:n])))
                actions.append(group.copy())
                offset += n
    x, ref, act = np.concatenate(hiddens), np.concatenate(logits), np.concatenate(actions)
    np.testing.assert_allclose(x @ params[kernel] + params[bias], ref, rtol=2e-4, atol=2e-4)
    return x, ref, act


def fit(x, logits, actions, w, b):
    # Whole 64-button blocks are reserved for internal model selection.
    held = (np.arange(len(actions)) // 64) % 5 == 4
    assert held.any() and (~held).any()
    data = tuple(jax.device_put(a) for a in (x, logits, actions.astype(np.int32)))
    original = weights = (jax.device_put(w), jax.device_put(b))
    optimizer = optax.adam(0.001)
    state = optimizer.init(weights)
    def loss(weights, inp, base_logits, labels):
        actual = inp @ weights[0] + weights[1]
        nll = optax.softmax_cross_entropy_with_integer_labels(actual, labels).mean()
        regularizer = (jax.nn.softmax(base_logits) *
                       (jax.nn.log_softmax(base_logits) - jax.nn.log_softmax(actual))).sum(-1).mean()
        return nll + regularizer, (nll, regularizer)
    @jax.jit
    def update(weights, state, indices):
        subset = [a[indices] for a in data]
        (_, _), grad = jax.value_and_grad(loss, has_aux=True)(weights, *subset)
        change, state = optimizer.update(grad, state)
        return optax.apply_updates(weights, change), state
    valid = [jax.device_put(a[held]) for a in (x, logits, actions.astype(np.int32))]
    assess = jax.jit(lambda weights: loss(weights, *valid)[1])
    baseline = float(jax.device_get(assess(original)[0]))
    best, best_loss, count = original, baseline, 0
    trajectory = []
    rng = np.random.default_rng(82019)
    train = np.flatnonzero(~held)
    for step in range(1,301):
        idx = jax.device_put(rng.choice(train, 128).astype(np.int32))
        weights, state = update(weights, state, idx)
        if step % 25 == 0:
            nll, kl = map(float, jax.device_get(assess(weights)))
            trajectory.append(dict(updates=step, heldout_nll=nll, heldout_kl=kl))
            if nll < best_loss and kl <= .15:
                best, best_loss, count = weights, nll, step
    return jax.device_get(best), dict(selected_updates=count, parent_nll=baseline,
        selected_nll=best_loss, compared_updates=300, validation_block_length=64,
        validation_decisions=int(held.sum()), fit_curve=trajectory)


def fit_checkpoint(dataset, parent, out, source):
    sys.path.insert(0, str(source))
    from embodied.jax import internal
    manifest = json.loads((dataset / "manifest.json").read_text())
    tape_id = manifest["source_sha256"]
    original, = (dataset / "audit").glob("*.bin")
    assert manifest["source_sha256"] == digest(original) == tape_id
    assert manifest["replay_restores_during_game"] == 0 and not manifest["frozen_evaluation_data_used"]
    out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((parent / "provenance.json").read_text())
    assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() == PIN
    subprocess.run(["git","-C",str(source),"diff","--exit-code","HEAD","--","dreamerv3","embodied"],check=True)
    frozen = parent / "frozen-policy"
    assert digest(frozen / "agent.pkl") == provenance["agent_sha256"]
    marker = out / "result.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if (prior["training_tape_sha256"] != tape_id or prior["parent_sha256"] != provenance["agent_sha256"]
                or prior["fitted_sha256"] != digest(out / "frozen-policy/agent.pkl")):
            raise ValueError("Output directory contains a different actor fit")
        return
    config = YAML(typ="safe").load((parent / "config.yaml").read_text())
    internal.setup(platform="cuda", compute_dtype="float32", prealloc=False, transfer_guard=False)
    devices = [d.device_kind for d in jax.devices() if d.platform == "gpu"]
    assert devices
    with (frozen / "agent.pkl").open("rb") as file:
        record = pickle.load(file)
    params = record["params"]
    x, raw, actions = features(params, dataset, config)
    assert actions.tobytes() == original.read_bytes()
    (w, b), result = fit(x, raw, actions, params[kernel], params[bias])
    params[kernel], params[bias] = w, b
    checkpoint = out / "frozen-policy"
    with tempfile.TemporaryDirectory(dir=out) as temp:
        stage = Path(temp) / "checkpoint"
        stage.mkdir()
        with (stage / "agent.pkl").open("wb") as file:
            pickle.dump(record, file, protocol=4)
        shutil.copyfile(frozen / "step.pkl", stage / "step.pkl")
        (stage / "done").touch()
        if checkpoint.exists():
            if digest(stage/"agent.pkl") != digest(checkpoint/"agent.pkl"):
                raise ValueError("Partial output contains different weights. Use a new output directory.")
        else:
            stage.rename(checkpoint)
    with (frozen / "agent.pkl").open("rb") as file:
        prior = pickle.load(file)
    changed = sorted(k for k,v in params.items() if not np.array_equal(v, prior["params"][k]))
    assert set(changed).issubset({kernel, bias}) and record["counters"] == prior["counters"]
    result.update(parent_training_steps=provenance["training_step"], parent_sha256=provenance["agent_sha256"],
        training_tape_sha256=tape_id, training_decisions=len(actions), changed_parameters=changed,
        evaluation_only_checkpoint=True, fitted_sha256=digest(checkpoint / "agent.pkl"), gpu=devices)
    interim = out / "result.tmp"
    interim.write_text(json.dumps(result, indent=2, allow_nan=False))
    interim.replace(marker)
    print(json.dumps({k:v for k,v in result.items() if k!="fit_curve"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_checkpoint(args.dataset.resolve(), args.source_run.resolve(), args.output.resolve(), args.upstream.resolve())
