"""Verify a clean original-ROM episode from boot and optionally render its real gameplay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image
from emerald import BUTTON_IDS, Emerald, REPEAT, ROM_SHA1, verify_rom
from pixel_env import Score, game_paths


def host_path(path,binary):
    if os.name!="nt" and binary.endswith(".exe"):
        return subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()
    return str(path)


def replay(audit,out,event,video):
    audit=audit.resolve()
    evidence=json.loads(audit.read_text())
    event_step=evidence["events"].get(event)
    if event_step is None:raise ValueError(f"{event} missing from episode")
    tape=audit.with_suffix(".bin").read_bytes()
    assert len(tape)==evidence["steps"]
    core, rom = game_paths()
    verify_rom(rom)
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    start=max(1,event_step-24)
    end=min(len(tape),event_step+10)
    writer=None
    temp=out/(event+".partial.mp4")
    final=out/(event+".mp4")
    if video:
        binary=shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not binary:raise FileNotFoundError("FFmpeg")
        writer=subprocess.Popen([binary,"-nostdin","-loglevel","error","-y","-f","rawvideo",
            "-pix_fmt","rgb24","-s","240x160","-r","16777216/280896","-i","pipe:0",
            "-vf","scale=960:640:flags=neighbor","-c:v","libx264","-pix_fmt","yuv420p",
            "-crf","16","-movflags","+faststart",host_path(temp,binary)],stdin=subprocess.PIPE)
    g=Emerald(str(core),str(rom))
    observed=Score()
    fight=badge_at=None
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
            if key[0]==6 and (read(0x02022fec,4)&8) and read(0x02038bca,2)==265:
                if fight is None:fight=n
                if n==fight+36:Image.fromarray(g.screen).save(out/"roxanne-battle.png")
            if key[0]==5 and g.verified_story(key) and key[1:3]==(11,3):
                sb=read(0x03005d8c,4)
                flags=lambda flag:read(sb+0x1270+flag//8,1)&(1<<(flag%8))
                if flags(0x4f0) and flags(0x867) and badge_at is None:badge_at=n
            if n==event_step:Image.fromarray(g.screen).save(out/(event+".png"))
        assert observed.events==evidence["events"],(observed.events,evidence["events"])
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
                input_sha256=hashlib.sha256(tape).hexdigest(),roxanne_battle=fight,stone_badge=badge_at)
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
    p.add_argument("--event",default="stone_badge")
    p.add_argument("--video",action="store_true")
    args=p.parse_args()
    replay(args.audit,args.out,args.event,args.video)
