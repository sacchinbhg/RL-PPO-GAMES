"""Run the official frozen evaluation, then probe its world model without updates."""
import json
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path[:0] = [str(project), "/opt/emerald-rl/source"]
from dreamerv3 import main as upstream
from embodied.jax import internal
import upstream_emerald

out = project / "runs/pixels-only-01/official-wsl/controls/heldout-world-model"


def probe(agent):
    with np.load(out/"evaluation-windows.npz") as dataset:
        observed = {k:dataset[k] for k in dataset.files}
    assert observed["action"].shape == (8,33)
    result = {}
    for treatment in ("recorded", "reversed"):
        parts = []
        # The upstream grid shows up to six columns. A second call supplies the final two.
        for run, (indices,n) in enumerate(((np.arange(8),6), (np.roll(np.arange(8),2),2))):
            inputs = agent._zeros(agent.spaces,(8,33))
            inputs.update({k:v[indices] for k,v in observed.items()})
            # Start with a cold model carry and observe the left context in the window.
            inputs["consec"][:] = 1
            if treatment == "reversed":
                inputs["action"][:,16:32] = inputs["action"][:,16:32][:,::-1].copy()
            data = internal.device_put(inputs, agent.train_sharded)
            data["seed"] = agent._seeds(9183+run, agent.train_mirrored)
            _, metrics = agent.report(agent.init_report(8), data)
            frames = metrics["openloop/image"][:,:,:68*n]
            truth = frames[:32].reshape(32,196,n,68,3).transpose(2,0,1,3,4)[:,:,2:66,2:66]
            assert np.array_equal(truth, observed["image"][indices[:n],1:])
            parts.append(frames)
        report = np.concatenate(parts,axis=2)
        path = out/(treatment+".npz")
        np.savez_compressed(path,frames=report)
        subprocess.run([sys.executable,str(Path(__file__).with_name("score-emerald-predictions.py")),
                        str(path), str(out/treatment)], check=True)
        result[treatment] = json.loads((out/treatment/"prediction-errors.json").read_text())["overall"]
    result["changed_buttons_fraction"] = float(np.mean(
        observed["action"][:,16:32] != observed["action"][:,16:32][:,::-1]))
    result["sample"] = "Eight fixed windows, at observation 2048 and 6144 in each of four frozen-policy games."
    result["checkpoint"] = 65436
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    print("Held-out world-model measurements", json.dumps(result), flush=True)


if __name__ == "__main__":
    make_agent = upstream.make_agent
    captured = []
    def record(config):
        agent = make_agent(config)
        captured.append(agent)
        return agent
    with patch.object(upstream,"make_agent",record):
        upstream_emerald.main()
    # A visualization error must not invalidate an otherwise completed policy evaluation.
    try:
        agent, = captured
        probe(agent)
    except Exception:
        out.mkdir(parents=True,exist_ok=True)
        (out/"probe-error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
