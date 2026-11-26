"""Wait for this experiment's first agent-discovered badge, fit a policy, and certify fresh wins."""
import json
import subprocess
import sys
import time
from gym_data import ROOT,RUN


def main():
    while True:
        winners=sorted(RUN.glob("*/tier-100-level-*.bin"),key=lambda p:p.stat().st_mtime_ns)
        if winners:break
        time.sleep(15)
    tape=winners[0]
    source=str(tape.relative_to(RUN))
    out="stone-policy"
    print("Training on agent-discovered badge route",source,flush=True)
    cmd=[sys.executable,"-u","gym_imitation.py","--source",source,"--out",out,
         "--steps","3200","--refine","3500","--pure"]
    subprocess.run(cmd,cwd=ROOT,check=True)
    for label in ("fresh","neural-only"):
        path=RUN/out/label
        report=json.loads((path/"evaluation.json").read_text())
        if "stone_badge" in report["events"]:
            subprocess.run([sys.executable,"-u","gym_verify.py",str(path)],cwd=ROOT,check=True)


if __name__=="__main__":
    main()
