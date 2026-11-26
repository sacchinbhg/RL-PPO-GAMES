"""Use separate native training seeds to earn starter games, then expand the highest-return training episode."""
import importlib.util
from pathlib import Path
import subprocess
import sys

work=Path(__file__).resolve().parent
project=work.parent/"outputs/EmeraldRL"
root=project/"runs/pixels-only-01/official-wsl"
out=root/"native-earned-openings"
sys.path.insert(0,str(project))
from earned_memory import Memory
from train_earned_dreamer import write_json
spec=importlib.util.spec_from_file_location("native_training",work/"explore-native-earned-screens.py")
native=importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

def main():
    out.mkdir(exist_ok=True)
    seeds=list(range(398821,398825))
    write_json(out/"plan.json",dict(seeds=seeds,decisions_per_game=131072,source="own RGB memory with ordinary fallback",
        training_only=True,heldout_games_used=False,choice="highest native scalar return across all four complete games; earliest seed breaks ties",
        expansion_new_decisions=1048576,expansion_seed=173744,backend="native port",fresh_original_verification_required=True))
    candidates=[]
    for seed in seeds:
        folder=out/f"seed-{seed}"
        folder.mkdir(exist_ok=True)
        audit=folder/"final/game.json"
        if not audit.exists():
            env=native.NativeGame(folder)
            policy=Memory(root/"earned-screen-memory/model",seed)
            try:
                previous=0
                frame=env.initial
                for i in range(1,131073):
                    choice=policy.choose(frame,previous/100)
                    policy.executed(choice)
                    frame,previous,done=env.step(choice)
                    if i%8192==0:
                        write_json(folder/"status.json",dict(buttons=i,earned_return_cents=env.total,events=env.events))
                    if done:
                        break
                env.record(audit.parent)
            finally:
                env.game.close()
        record=native.read(audit)
        assert record["training_only"] and (record["steps"]==131072 or "stone_badge" in record["events"])
        candidates.append((record["return_cents"],-seed,audit))
    value,seed,audit=max(candidates)
    write_json(out/"selected.json",dict(seed=-seed,return_cents=value,tape=native.read(audit)["tape_sha256"],
                                      input=str(audit),actual_training_decisions=sum(native.read(row[2])["steps"] for row in candidates)))
    subprocess.run([sys.executable,"-u",str(work/"explore-native-earned-screens.py"),"--steps","1048576","--seed","173744",
                    "--earned-native",str(audit)],check=True)

if __name__=="__main__":
    main()
