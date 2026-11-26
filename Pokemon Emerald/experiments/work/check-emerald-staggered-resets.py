"""Verify truncation lengths, resets, and actor-only frozen-evaluation rules on the real adapter."""
import json
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
out = project / "runs/pixels-only-01/official-wsl/controls/staggered-reset-smoke"
out.mkdir(exist_ok=True)
base = [sys.executable, "-u", str(project/"upstream_emerald.py"), "--upstream", "/opt/emerald-rl/source",
        "--logdir", str(out/"train"), "--platform","cuda","--envs","4","--batch","4","--length","8",
        "--episode","64","--short-episode-envs","2","--short-episode-length","16",
        "--screen-exploration-envs","2","--parallel-envs","--seed","22911","--steps","264"]
with (out/"train.log").open("w") as file:
    subprocess.run(base,stdout=file,stderr=subprocess.STDOUT,check=True)
training = json.loads((out/"train/provenance.json").read_text())
assert training["worker_episode_lengths"] == [64,64,16,16]
counts = []
for worker, length in enumerate(training["worker_episode_lengths"]):
    tapes = list((out/"train/external-audit").glob(f"*/env-{worker}/*.bin"))
    assert len(tapes) == 66//(length+1) and all(len(p.read_bytes())==length for p in tapes)
    counts.append(len(tapes))
bad = subprocess.run(base+["--evaluate",str(out/"train/ckpt")],capture_output=True,text=True)
assert bad.returncode and "short episodes require training" in bad.stderr
(out/"verification.json").write_text(json.dumps(dict(lengths=training["worker_episode_lengths"],
    finished_episodes=counts, frozen_evaluation_rejected_short_episodes=True, platform="cuda"),indent=2))
