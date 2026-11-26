"""Hash every control decision and observation across the small stock and reduced-rendering fork trials."""
import hashlib
from pathlib import Path
import struct
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
sys.path.insert(0,str(base/"work/renderer-experimental"))
import explore_reward_frontiers as experiment
from reward_branch_env import ForkEnv
from reward_branch_search import write_json

class RecordingFork(ForkEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stream=hashlib.sha256(self.initial)
        self.count=0
        self.restores=0

    def command(self,name,value):
        result=super().command(name,value)
        self.stream.update(name.encode())
        if name=="step":
            image,cents,done,used=result
            self.stream.update(struct.pack("<BBi?B",*value,cents,done,used))
            self.stream.update(image)
            self.count+=used
        elif name in ("save","restore"):
            self.stream.update(struct.pack("<I",value))
            if name=="restore":
                self.stream.update(result)
                self.restores+=1
        return result

    def close(self):
        super().close()
        write_json(Path(sys.argv[1])/"stream.json",dict(sha256=self.stream.hexdigest(),emulated_decisions=self.count,
                   restored_screen_count=self.restores))

if __name__=="__main__":
    dest=Path(sys.argv[1])
    root=project/"runs/pixels-only-01/official-wsl"
    source=root/"controls/outcome-collection-smoke"
    experiment.ForkEnv=RecordingFork
    experiment.run(dest,source,source/"best.json",root/"controls/visual-motion/route-training.json",512,173741,16,
                   sys.argv[2]=="reduced")
