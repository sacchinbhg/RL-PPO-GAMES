"""Reload two idle shell queues before they start the next experiment."""
import os
from pathlib import Path
import signal
import subprocess
import time

jobs={"work/run-emerald-earned-long-pair.sh","work/run-emerald-heldout-earned.sh"}
for script in sorted(jobs):
    matches=[]
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command=(proc/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        except (FileNotFoundError,PermissionError,UnicodeDecodeError,ProcessLookupError):
            continue
        if command==["bash",script]:
            matches.append(proc)
    if len(matches)!=1:
        raise RuntimeError((script,matches))
    proc,=matches
    pid=int(proc.name)
    children=(proc/f"task/{pid}/children").read_text().split()
    if not children or any((Path("/proc")/child/"comm").read_text().strip()!="sleep" for child in children):
        raise RuntimeError(f"{script} is running a workload. Its current process has been preserved.")
    os.kill(pid,signal.SIGTERM)
    for child in children:
        try:
            os.kill(int(child),signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(30):
        if not proc.exists():
            break
        time.sleep(.1)
    if proc.exists():
        raise RuntimeError(f"The prior idle queue {pid} is still active")
    subprocess.Popen(["bash",script],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print("Reloaded idle queue:",script,flush=True)
