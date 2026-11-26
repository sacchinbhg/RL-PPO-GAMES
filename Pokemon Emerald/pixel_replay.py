"""Verify a clean original-ROM episode from boot and optionally render its real gameplay."""
import argparse
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

from PIL import Image
from emerald import BUTTON_IDS, Emerald, MAPS, REPEAT, ROM_SHA1, verify_rom
from pixel_env import Score, game_paths


STARTERS = {277: "Treecko", 280: "Torchic", 283: "Mudkip"}
GROWTH_BLOCK = tuple(order.index(0) for order in permutations(range(4)))


def verify_starter(env, decision):
    """Independently identify the first party member and the game's starter-choice variable."""
    mon = env.ram(0x020244ec, 100)
    personality, trainer_id = struct.unpack_from("<II", mon)
    key = personality ^ trainer_id
    secure = struct.pack("<12I", *(word ^ key for word in struct.unpack_from("<12I", mon, 32)))
    checksum, = struct.unpack_from("<H", mon, 28)
    if sum(struct.unpack("<24H", secure)) % 65536 != checksum:
        raise ValueError("First party member has an invalid checksum")
    species, = struct.unpack_from("<H", secure, 12 * GROWTH_BLOCK[personality % 24])
    save = int.from_bytes(env.ram(0x03005d8c, 4), "little")
    choice = int.from_bytes(env.ram(save + 0x139c + 2 * 0x23, 2), "little")
    if choice >= 3 or species != (277, 280, 283)[choice]:
        raise ValueError(f"Party species {species} does not match the starter choice {choice}")
    count = env.ram(0x020244e9, 1)[0]
    if not 1 <= count <= 6 or not 5 <= mon[84] <= 100:
        raise ValueError("Invalid party count or level at starter acquisition")
    return dict(decision=decision, species_id=species, species_name=STARTERS[species],
                level=mon[84], party_count=count, checksum=checksum)


def host_path(path,binary):
    if os.name!="nt" and binary.endswith(".exe"):
        return subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()
    return str(path)


def replay(audit,out,event,video,map_name=None,decision=None):
    audit=audit.resolve()
    evidence=json.loads(audit.read_text())
    if decision is not None:
        if event is not None or map_name is not None or not 1<=decision<=evidence["steps"]:
            raise ValueError("A review decision must be in the recorded game and used on its own")
        event_step=decision
        label=f"decision-{decision}"
    elif map_name is not None:
        if not any(map_name in group for group in MAPS):
            raise ValueError(f"Unknown Emerald map: {map_name}")
        event_step=evidence.get("maps_first_reached",{}).get(map_name)
        label="map-"+map_name
    else:
        event_step=evidence["events"].get(event)
        label=event
    if event_step is None:raise ValueError(f"{label} missing from episode")
    tape=audit.with_suffix(".bin").read_bytes()
    assert len(tape)==evidence["steps"]
    core, rom = game_paths()
    verify_rom(rom)
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    start=max(1,event_step-24)
    end=min(len(tape),event_step+10)
    writer=None
    temp=out/(label+".partial.mp4")
    final=out/(label+".mp4")
    if video:
        binary=shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not binary:raise FileNotFoundError("FFmpeg")
        writer=subprocess.Popen([binary,"-nostdin","-loglevel","error","-y","-f","rawvideo",
            "-pix_fmt","rgb24","-s","240x160","-r","16777216/280896","-i","pipe:0",
            "-vf","scale=960:640:flags=neighbor","-c:v","libx264","-pix_fmt","yuv420p",
            "-crf","16","-movflags","+faststart",host_path(temp,binary)],stdin=subprocess.PIPE)
    g=Emerald(str(core),str(rom))
    observed=Score(evidence.get("uncapped_exploration", False))
    fight=badge_at=starter=None
    read=lambda a,n:int.from_bytes(g.ram(a,n),"little")
    try:
        g.load(g.boot)
        for n,a in enumerate(tape,1):
            if writer and start<=n<=end:
                button=BUTTON_IDS[a]
                for frame in range(REPEAT):
                    g.buttons=(1<<button) if button is not None and frame<14 else 0
                    g.capture=True
                    g.lib.retro_run()
                    writer.stdin.write(g.screen.tobytes())
                key=g.descriptor()
            else:_,key=g.step(a)
            observed.read(g,key)
            if n == evidence["events"].get("starter"):
                starter = verify_starter(g, n)
            if key[0]==6 and (read(0x02022fec,4)&8) and read(0x02038bca,2)==265:
                if fight is None:fight=n
                if n==fight+36:Image.fromarray(g.screen).save(out/"roxanne-battle.png")
            if key[0]==5 and g.verified_story(key) and key[1:3]==(11,3):
                sb=read(0x03005d8c,4)
                flags=lambda flag:read(sb+0x1270+flag//8,1)&(1<<(flag%8))
                if flags(0x4f0) and flags(0x867) and badge_at is None:badge_at=n
            if n==event_step:Image.fromarray(g.screen).save(out/(label+".png"))
        assert observed.events==evidence["events"],(observed.events,evidence["events"])
        if "maps_first_reached" in evidence:
            assert observed.first_maps == evidence["maps_first_reached"]
        if map_name is not None:
            assert observed.first_maps.get(map_name) == event_step
        if event=="stone_badge":
            assert fight and badge_at and fight<badge_at and badge_at==event_step
    finally:
        g.close()
        if writer:
            writer.stdin.close()
            if writer.wait()!=0:raise RuntimeError("FFmpeg encoding failed")
    result=dict(source_audit=str(audit),checked_event=event,verified_decision=event_step,
                verified_events=observed.events,buttons=len(tape),fresh_boot=True,restores_during_game=0,
                rom_sha1=ROM_SHA1,core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
                input_sha256=hashlib.sha256(tape).hexdigest(),roxanne_battle=fight,stone_badge=badge_at,
                starter=starter,maps_first_reached=observed.first_maps)
    if map_name is not None:
        result["checked_map"]=map_name
    if decision is not None:
        result["reviewed_decision"]=decision
    if video:
        binary=shutil.which("ffprobe") or shutil.which("ffprobe.exe")
        meta=json.loads(subprocess.check_output([binary,"-v","error","-show_entries",
            "stream=nb_frames,width,height,codec_name","-of","json",host_path(temp,binary)],text=True))
        measured=meta["streams"][0]
        assert int(measured["nb_frames"])==(end-start+1)*REPEAT
        temp.replace(final)
        result.update(clip_from=start,clip_through=end,video=measured)
    (out/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("audit",type=Path)
    p.add_argument("out",type=Path)
    goal=p.add_mutually_exclusive_group()
    goal.add_argument("--event")
    goal.add_argument("--map",dest="map_name",help="Verify the first original-ROM entry into a recorded map")
    goal.add_argument("--decision",type=int,help="Capture a chosen button decision while replaying the recorded game")
    p.add_argument("--video",action="store_true")
    args=p.parse_args()
    default_event=None if args.map_name is not None or args.decision is not None else "stone_badge"
    replay(args.audit,args.out,args.event or default_event,args.video,args.map_name,args.decision)
