"""Compare normal and delayed-input fresh games with the same frozen self-imitation model."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import fit


def main():
    root=project/"runs/pixels-only-01/official-wsl"
    family=root/"earned-screen-memory"
    family.mkdir(exist_ok=True)
    trees=root/"reward-only-training-forks/verified-training-tapes/24479561ff0b9fe07b3932aeb611e70d3b7b7f4b42a9f6fe428cea84e19fa412"
    fire=root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes/e30822c294f0d5469126f3e250c9ad4ac83b232534b37d6f54b24b56078e37c5"
    with (family/"testing.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=family/"model"
        fit([trees,fire],model)
        processes=[]
        for baseline,label,delay in (("treecko","clean",0),("two-earned-games","perturbed",257)):
            complete=root/"earned-visual-model"/baseline/"fresh-seed-24917-limit-65536/result.json"
            while not complete.exists():
                time.sleep(10)
            out=family/label
            if (out/"result.json").exists():
                continue
            command=[sys.executable,"-u",str(project/"earned_memory.py"),"--dataset",str(trees),
                     "--dataset",str(fire),"--model",str(model),"--output",str(out),
                     "--limit","65536","--wait-every",str(delay),"--seed","197461"]
            with (family/f"{label}.log").open("a") as log:
                processes.append((label,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)))
        for label,p in processes:
            if p.wait()!=0:
                raise RuntimeError(f"The {label} test failed")
        print(json.dumps({s:json.loads((family/s/"result.json").read_text()) for s in ("clean","perturbed")}),flush=True)


if __name__=="__main__":
    main()
