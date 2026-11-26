"""Verify public experiment entry points and the stored provenance of the live comparison."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import verified_reset_lineage
from explore_earned_endpoints import run
from learn_persistent_effects import compare
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
read=lambda p:json.loads(p.read_text())

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    folder=root/"controls/endpoint-button-sweep"
    original=folder/"smoke"
    new=folder/"public-smoke"
    source=[family/("bank-movement-"+n) for n in ("ordinary","learned")]
    run(source,root/"controls/dreamer-earned-resets/starts",new,1,[0],[4],173751)
    current=read(new/"completed.json")
    assert current==read(original/"completed.json")
    rows=[read(p) for p in new.glob("discoveries/work-*/record.json")]
    assert sorted(r["first_button"] for r in rows)==list(range(9))
    for r in rows:
        assert verified_reset_lineage(new,read(new/"settings.json"),(new/r["audit"]/"game.bin").read_bytes())
    for dest in (original,family/"every-sealed-endpoint-01"):
        config=read(dest/"settings.json")
        origin=read(dest/"branch-origin.json")
        if "source_endpoint_search_buttons" not in origin:
            assert origin["source_run"]==read(source[0]/"branch-origin.json")["source_run"]
            config["branch_count_scope"]="this endpoint experiment"
            write_json(dest/"settings.json",config)
            write_json(dest/"branch-origin.json",dict(origin,seed=config["seed"],inherited_training_buttons=origin["parent_record"]["work"]+config["inherited_new_work"],
                       source_endpoint_search_buttons=config["inherited_new_work"],source_endpoint_arms=config["source_arms"]))
    effects=root/"controls/persistent-button-effects"
    training=next((family/"visual-slots-0/verified-training-tapes").glob("e30822*"))
    heldout=next((root/"earned-frontier-curriculum/verified-training-tapes").glob("01f5c6*"))
    with tempfile.TemporaryDirectory() as tmp:
        draft=Path(tmp)
        for n in ("train","heldout"):
            (draft/n).mkdir()
            for f in ("registration.json","examples.npz","complete.json"):
                shutil.copyfile(effects/n/f,draft/n/f)
        compare(training,heldout,draft)
        assert read(draft/"result.json")==read(effects/"result.json")
    write_json(folder/"public-verified.json",dict(real_endpoint_actions=current["status"]["new_buttons"],distinct_first_buttons=9,
               all_input_histories_verified=True,matched_reference_search=True,independently_refit_effect_model_and_metrics=True,
               effect_model=read(effects/"result.json")["model_sha256"]))
