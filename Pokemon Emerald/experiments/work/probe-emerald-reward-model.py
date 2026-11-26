"""Read-only WSL GPU measurement on frozen weights and held-out scenes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import elements
import jax
from ruamel.yaml import YAML

sys.path.insert(0, "/opt/emerald-rl/source")
from dreamerv3 import main as upstream
from dreamerv3.agent import Agent
from embodied.jax import internal

root = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out = root / "controls/heldout-rewards"


class Spaces:
    obs_space = dict(image=elements.Space(np.uint8,(64,64,3)),reward=elements.Space(np.float32),
                     is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
    act_space = dict(action=elements.Space(np.int32,(),0,9),reset=elements.Space(bool))
    def close(self):
        pass


def reward_report(self, initial, data):
    carry, obs, action, _ = self._apply_replay_context(initial, data)
    split = lambda xs, span: jax.tree.map(lambda x: x[:,span], xs)
    earlier, final = slice(None,-1), slice(-1,None)
    past, current = split(obs,earlier), split(obs,final)
    before, last_button = split(action,earlier), split(action,final)
    enc, _, tokens = self.enc(carry[0], past, past["is_first"], training=False)
    dyn, _, _ = self.dyn.observe(carry[1], tokens, before, past["is_first"], training=False)
    _, prior, _ = self.dyn.imagine(dyn, last_button, 1, training=False)
    _, _, token = self.enc(enc, current, current["is_first"], training=False)
    _, _, posterior = self.dyn.observe(dyn, token, last_button, current["is_first"], training=False)
    predict = lambda feat: self.rew(self.feat2tensor(feat),2).pred()[:,-1]
    return initial, dict(prior=predict(prior), posterior=predict(posterior))


def read_data():
    rows, ims, acts = [], [], []
    for game in json.loads((out/"sources.json").read_text()):
        with np.load(out/(game["env"]+".npz")) as ds:
            for idx, row in enumerate(game["windows"]):
                assert abs(float(ds["reward"][idx,-1]) - row["reward"]) < 1e-4
                rows.append(dict(game=game["env"], **row))
                ims.append(ds["image"][idx])
                acts.append(ds["action"][idx])
    return rows, np.stack(ims), np.stack(acts)


def run():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("phase")
    parser.add_argument("--adapted",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    arm,phase=args.arm,args.phase
    evaluation = root / arm / phase
    provenance = json.loads((evaluation/"provenance.json").read_text())
    source = "/opt/emerald-rl/source"
    revision = subprocess.check_output(["git","-C",source,"rev-parse","HEAD"],text=True).strip()
    assert revision == provenance["upstream_revision"]
    subprocess.run(["git","-C",source,"diff","--exit-code","HEAD","--","dreamerv3","embodied"],check=True)
    frozen = evaluation/"frozen-policy"
    fit=None
    sha=provenance["agent_sha256"]
    if args.adapted:
        fit=json.loads((args.adapted/"result.json").read_text())
        assert fit["parent_sha256"]==sha and fit["upstream_revision"]==revision and args.output
        frozen=args.adapted/"frozen-policy"
        sha=fit["trained_sha256"]
    with (frozen/"agent.pkl").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == sha
    rows, images, actions = read_data()
    raw = YAML(typ="safe").load((evaluation/"config.yaml").read_text())
    raw["jax"]["precompile"] = False
    assert raw["jax"]["platform"] == "cuda"
    with patch.object(upstream,"make_env",lambda *args, **kwargs: Spaces()), patch.object(Agent,"report",reward_report):
        agent = upstream.make_agent(elements.Config(raw))
        devs = [x.device_kind for x in jax.devices() if x.platform == "gpu"]
        assert devs
        cp = elements.Checkpoint()
        cp.agent = agent
        cp.load(str(frozen), keys=["agent"])
        updates = int(agent.n_updates)
        batch = raw["batch_size"]
        def measure(idx, seed, blank=False):
            inputs = agent._zeros(agent.spaces,(batch,33))
            inputs["image"] = images[idx].copy()
            if blank:
                inputs["image"][:,-1] = 0
            inputs["action"] = actions[idx]
            inputs["consec"][:] = 1
            assert not inputs["reward"].any()
            data = internal.device_put(inputs, agent.train_sharded)
            data["seed"] = agent._seeds(seed, agent.train_mirrored)
            _, metrics = agent.report(agent.init_report(batch), data)
            return {key:np.asarray(metrics[key]) for key in ("prior","posterior")}
        for start in range(0,len(rows),batch):
            idx = np.minimum(np.arange(start,start+batch),len(rows)-1)
            draws = [measure(idx, 10487+n) for n in range(4)]
            if start == 0:
                blank = measure(idx,10487,blank=True)
                assert np.array_equal(draws[0]["prior"],blank["prior"])
            for pos in range(min(batch,len(rows)-start)):
                for key in ("prior","posterior"):
                    rows[start+pos][key] = float(np.mean([x[key][pos] for x in draws]))
        assert int(agent.n_updates) == updates
    output = dict(arm=arm, phase=phase, checkpoint=sha, upstream_revision=revision,
                  training_steps=provenance["training_step"], gpu=devs, windows=rows,
                  prior_invariant_to_final_image=True, scalar_reward_input_zero=True,
                  stochastic_draws=4, optimizer_updates_during_probe=0)
    for key in ("prior","posterior"):
        pairs = []
        for row in rows:
            if row["offset"]:
                continue
            controls = [c[key] for c in rows if c["offset"] and c["game"]==row["game"] and c["event"]==row["event"]]
            pairs.extend([(row[key]>v)+0.5*(row[key]==v) for v in controls])
        output[key] = dict(mean_on_story=float(np.mean([x[key] for x in rows if not x["offset"]])),
                           mean_on_nearby_zero=float(np.mean([x[key] for x in rows if x["offset"]])),
                           pairwise_ranking=float(np.mean(pairs)), control_pairs=len(pairs))
    if fit:
        output["additional_optimizer_updates"]=fit["optimizer_updates"]
        output["training_tapes"]=fit["training_tapes"]
    dest=args.output or out/(arm+"-"+phase+".json")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,indent=2,allow_nan=False))
    print(arm,phase,output["prior"],output["posterior"],flush=True)


if __name__ == "__main__":
    run()
