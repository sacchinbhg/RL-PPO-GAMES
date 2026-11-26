"""Seal the fit using training windows first, then score its choices on the pre-existing independent trials."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from calibrate_button_forecasts import context_hash,fit,score
from train_earned_dreamer import digest,write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"earned-button-alternatives"
training=out/"training"
external=root/"controls/heldout-action-rewards"
read=lambda p:json.loads(p.read_text())

def available():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))

def evaluate():
    meta=read(out/"calibration/fit.json")
    assert not meta["evaluation_data_used"] and digest(out/"calibration/model.npz")==meta["model_sha256"]
    contexts,past,actual,group=[],[],[],[]
    for file in sorted(external.glob("env-*.npz")):
        with np.load(file) as ds:
            contexts.extend(ds["context"]); past.extend(ds["past"]); actual.extend(ds["real_rewards"].sum(-1))
            group.extend([file.stem]*len(ds["context"]))
    reference=read(external/"earned-1152-ranking.json")
    with np.load(external/"earned-1152-forecasts.npz") as f:
        forecasts=f["forecasts"]
    assert reference["weights"]==meta["weights"] and len(forecasts)==len(actual)==len(group)==39
    actual=np.asarray(actual)
    group=np.asarray(group)
    with np.load(out/"calibration/model.npz") as model:
        training_hashes=set(model["training_context_hashes"].tolist())
        holdout=np.asarray([context_hash(im,btn) not in training_hashes for im,btn in zip(contexts,past)])
        estimates=score(model,np.asarray(contexts),forecasts)
    selected=holdout & (np.ptp(actual,axis=1)>.001)
    assert selected.any()
    results={}
    for name,predicted in estimates.items():
        chosen=np.isclose(predicted,predicted.max(1,keepdims=True),rtol=0,atol=1e-8)
        realized=(actual*chosen).sum(1)/chosen.sum(1)
        all_pairs=actual[:,:,None]>actual[:,None,:]+.001
        better=predicted[:,:,None]>predicted[:,None,:]
        equal=predicted[:,:,None]==predicted[:,None,:]
        pairs=all_pairs[selected]
        results[name]=dict(actual_mean=float(realized[selected].mean()),
            accurate_pair_fraction=float((better[selected][pairs]+.5*equal[selected][pairs]).mean()),
            by_game={g:float(realized[selected & (group==g)].mean()) for g in sorted(set(group[selected]))})
    if holdout.all():
        sensitive=[r for r in reference["records"] if r["action_sensitive_pairs"]]
        assert len(sensitive)==selected.sum()
        expected=np.mean([r["real_return_at_model_choice"] for r in sensitive])
        assert abs(results["world_model"]["actual_mean"]-expected)<1e-5
    uniform=float(actual[selected].mean())
    combined=results["calibrated_world_model"]["actual_mean"]
    strongest=max(results[k]["actual_mean"] for k in ("action_only","similar_screens","world_model"))
    write_json(out/"heldout-calibration.json",dict(registered_model=meta,all_windows=len(actual),
        exact_training_contexts_excluded=int((~holdout).sum()),heldout_sensitive_windows=int(selected.sum()),
        uniform_baseline=uniform,best_possible=float(actual[selected].max(1).mean()),results=results,
        qualifies_for_independent_replication=combined>uniform and combined>strongest))

def run():
    write_json(out/"calibration-plan.json",dict(forecast_model="frozen 1152-update official Dreamer",
        training="two sealed self-earned games, all action-sensitive contexts, no held-out outcomes",
        features="four centered imagined reward bins and a bias per button",ridge_strength=.1,
        comparators=["uniform","best button on own training data","four nearest observed contexts","uncalibrated official forecast"],
        policy_change="only a result above all baselines on independent replications could justify a live planning experiment"))
    while not (out/"verified.json").exists():
        time.sleep(30)
    while not (training/"world-model-forecasts.json").exists():
        if available()<2750000:
            time.sleep(30)
            continue
        with (root/"controls/large-optimizer.lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(30)
                continue
            if available()<2750000:
                continue
            with (out/"calibration.log").open("a") as log:
                subprocess.run([sys.executable,"-u",str(project/"forecast_button_alternatives.py"),
                    "--training",str(training),"--parent",str(root/"seed-71299-h15/eval-65536"),
                    "--checkpoint",str(root/"earned-official-updates/adaptation/frozen-policy"),
                    "--upstream","/opt/emerald-rl/source"],stdout=log,stderr=subprocess.STDOUT,check=True)
    fit(training,out/"calibration")
    evaluate()

if __name__=="__main__":
    os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false")
    out.mkdir(exist_ok=True)
    with (out/"calibration.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
