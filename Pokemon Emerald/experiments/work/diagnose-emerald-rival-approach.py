"""Human-only review of a completed tape. No information travels to a training or gameplay process."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald, MAPS, ROM_SHA1, verify_rom
from pixel_env import Score, game_paths
from reward_branch_search import write_json


def review(audit, out):
    original=json.loads(audit.read_text())
    tape=audit.with_suffix(".bin").read_bytes()
    assert len(tape)==original["steps"]
    data=json.loads((base/"work/pokeemerald/data/maps/Route103/map.json").read_text())
    rival,=[event for event in data["object_events"] if event.get("local_id")=="LOCALID_ROUTE103_RIVAL"]
    assert rival["movement_range_x"]==rival["movement_range_y"]==0
    template=(rival["x"]+7,rival["y"]+7)
    vectors={1:(0,1),2:(0,-1),3:(-1,0),4:(1,0)}
    core,rom=game_paths()
    verify_rom(rom)
    game=Emerald(str(core),str(rom))
    counts=Counter()
    distances=Counter()
    nearest=None
    try:
        game.load(game.boot)
        key=game.descriptor()
        score=Score(original["uncapped_exploration"])
        for n,button in enumerate(tape,1):
            if key[0]==5 and game.verified_story(key) and MAPS[key[1]][key[2]]=="Route103":
                counts["overworld_decision_boundaries"]+=1
                avatar=game.ram(0x02037590,8)
                objects=game.ram(0x02037350,0x24*16)
                slots=[objects[i:i+0x24] for i in range(0,len(objects),0x24)]
                if avatar[5]<len(slots):
                    player=slots[avatar[5]]
                    if player[0]&1 and player[2]&1:
                        counts["valid_player_samples"]+=1
                        x,y=struct.unpack_from("<hh",player,0x10)
                        dx,dy=template[0]-x,template[1]-y
                        distance=abs(dx)+abs(dy)
                        distances[distance]+=1
                        if nearest is None or distance<nearest[0]:
                            nearest=(distance,n)
                        matches=[s for s in slots if s[0]&1 and s[8]==2 and s[9]==key[2] and s[10]==key[1]]
                        if matches:
                            target,=matches
                            counts["rival_object_loaded"]+=1
                            if struct.unpack_from("<hh",target,0x10)==template:
                                counts["live_rival_matched_map_template"]+=1
                                if distance==1:
                                    counts["adjacent_to_loaded_rival"]+=1
                                    if vectors.get(player[0x18]&15)==(dx,dy):
                                        counts["facing_adjacent_rival"]+=1
                                        if button==1:
                                            counts["a_issued_while_facing_adjacent_rival"]+=1
            _,key=game.step(button)
            score.read(game,key)
        assert score.events==original["events"] and len(score.cells)==original["first_visit_cells"]
        if counts["overworld_decision_boundaries"] and nearest is None:
            raise ValueError("Route 103 appeared but no player object could be validated")
        out.parent.mkdir(parents=True,exist_ok=True)
        write_json(out,dict(input_sha256=hashlib.sha256(tape).hexdigest(),buttons=len(tape),rom_sha1=ROM_SHA1,
                   nearest_template_manhattan_tiles=nearest[0] if nearest else None,
                   first_button_at_nearest=nearest[1] if nearest else None,
                   first_maps=score.first_maps,events=score.events,first_visit_cells=len(score.cells),
                   counts=dict(counts),distance_histogram=dict(sorted(distances.items())),
                   camera_sampling="once immediately before each 24-frame button decision",in_game_restores=0,
                   destination="human diagnostics only"))
    finally:
        game.close()


if __name__=="__main__":
    review(Path(sys.argv[1]),Path(sys.argv[2]))
