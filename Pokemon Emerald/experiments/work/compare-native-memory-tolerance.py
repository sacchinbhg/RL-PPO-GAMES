"""Ablate approximate screen matching after detecting a repeated false match in an independent native game."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

work=Path(__file__).resolve().parent
project=work.parent/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import write_json
spec=importlib.util.spec_from_file_location("transfer",work/"test-native-visual-transfer.py")
transfer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
out=transfer.root/"controls/native-memory-tolerance"

if __name__=="__main__":
    out.mkdir(exist_ok=True)
    transfer.out=out
    seeds=list(range(197470,197474))
    write_json(out/"registration.json",dict(seeds=seeds,buttons_per_game=65536,palette="native RGB quantized to original 5-bit channels",
        primary="fresh games exiting the truck, getting a starter and reaching later maps",
        controls=["ordinary approximate and exact retrieval","exact-only retrieval","uniform buttons"],
        origin="false approximate matches occupied over 64900 decisions in each prior perturbed-rendering game",
        reference="all reported native results require a separate original-ROM test"))
    results={mode:[] for mode in ("ordinary","exact","uniform")}
    for seed in seeds:
        for mode in results:
            name=f"{mode}-{seed}"
            marker=out/name/"result.json"
            if not marker.exists():
                transfer.play(name,np.asarray([248,248,248],np.uint8),seed,mode=="uniform",mode!="exact")
            r=transfer.native.read(marker)
            results[mode].append(dict(seed=seed,starter=r["events"].get("starter"),town=r["maps_first_reached"].get("LittlerootTown"),
                maps=list(r["maps_first_reached"]),buttons=r["steps"],input_sha256=r["tape_sha256"]))
        write_json(out/"status.json",dict(finished_seeds=len(results["ordinary"]),games=results))
    write_json(out/"result.json",dict(arms=results))
