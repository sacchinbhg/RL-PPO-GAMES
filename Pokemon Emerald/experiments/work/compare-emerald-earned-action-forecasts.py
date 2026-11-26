"""Test whether offline Dreamer updates improve the first-button choice before deploying any planner."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
root=base/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out=root/"controls/heldout-action-rewards"
read=lambda p:json.loads(p.read_text())

def write(name,data):
    draft=out/(name+".tmp")
    draft.write_text(json.dumps(data,indent=2))
    draft.replace(out/(name+".json"))

def available():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))

def run():
    arms={"ordinary-65536":None,"earned-1152":root/"earned-official-updates/adaptation",
          "uncapped-earned-1152":root/"earned-official-updates/uncapped-training/adaptation"}
    write("earned-comparison-plan",dict(arms=list(arms),same_fresh_windows=39,first_button_alternatives=9,
        imagination_horizon=15,random_draws_per_action=8,source="existing held-out early-game counterfactuals",
        primary="mean actual original-game return at the model-selected action on windows sensitive to the choice",
        deploy="a positive result would need replication on new windows before using forecasts for gameplay"))
    results={}
    for label,fit in arms.items():
        target=out/f"{label}-ranking.json"
        while not target.exists():
            survival=root/"outcome-cost-comparison"
            if (survival/"input-checks.json").exists() and not (survival/"loss-cost/fit/result.json").exists():
                write("earned-comparison-status",dict(time=time.time(),waiting_for="paired survival training",arm=label))
                time.sleep(30)
                continue
            if available()<2800000:
                write("earned-comparison-status",dict(time=time.time(),waiting_for="guest RAM",arm=label,available_kib=available()))
                time.sleep(30)
                continue
            with (root/"controls/large-optimizer.lock").open("a") as gate:
                try:
                    fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(30)
                    continue
                if available()<2800000:
                    continue
                args=[sys.executable,"-u",str(base/"work/score-emerald-action-reward-forecasts.py"),"seed-71299-h15",
                      "--phase","eval-65536","--label",label]
                if fit:
                    args+=["--adapted",str(fit)]
                write("earned-comparison-status",dict(time=time.time(),measuring=label))
                with (out/"earned-comparison.log").open("a") as log:
                    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=read(target)
        assert data["optimizer_updates"]==0 and data["sensitive_windows"]==13 and data["horizon"]==15
        assert data["real_reward_input_zero"] and data["future_image_invariance_verified"]
        chosen=[r for r in data["records"] if r["action_sensitive_pairs"]]
        results[label]={k:data[k] for k in ("weights","actual_choice_mean","actual_uniform_mean","actual_oracle_mean","correct_pairs","comparable_pairs")}
        results[label]["reported_means_over_all_windows"]=len(data["records"])
        results[label]["sensitive_windows"]=len(chosen)
        results[label]["sensitive_window_returns"]={k:sum(r[v] for r in chosen)/len(chosen) for k,v in
                (("model_choice","real_return_at_model_choice"),("uniform","real_return_uniform"),("best_possible","real_return_best"))}
    write("earned-comparison",results)

if __name__=="__main__":
    os.environ.update(JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    with (out/"earned-comparison.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
