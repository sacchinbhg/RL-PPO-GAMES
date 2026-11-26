"""Match all eight original-game action histories to the two frozen 131072-decision evaluations."""
import json
from pathlib import Path
import sys

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import digest,write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"earned-official-updates/long-pair"
read=lambda p:json.loads(p.read_text())
if __name__=="__main__":
    rows={}
    for name in ("adapted","unchanged"):
        arm=out/name
        identity=read(arm/"result.json")
        assert identity["decisions_per_game"]==131072 and identity["games"]==4 and identity["seed"]==271342
        assert identity["in_game_restores"]==identity["weight_updates"]==0
        games=[]
        for lane in range(4):
            q=arm/"games"/f"env-{lane}"
            audit=next(q.glob("step-0000131072/*episode-00001.json"),None)
            if audit is None:
                audit=max(q.glob("step-*/*episode-00001.json"),key=lambda p:read(p)["steps"])
            data=read(audit)
            tape=audit.with_suffix(".bin")
            assert data["steps"]==tape.stat().st_size==131072
            games.append(dict(lane=lane,events=data["events"],maps=data["maps_first_reached"],input_sha256=digest(tape)))
        rows[name]=dict(weights=identity["weight_sha256"],games=games,starters=sum("starter" in g["events"] for g in games),
                        lab_handoffs=sum("lab_handoff" in g["events"] for g in games))
    write_json(out/"comparison.json",dict(seed=271342,decisions_per_game=131072,fresh_boot=True,
               in_game_restores=0,additional_weight_updates_during_games=0,arms=rows))
