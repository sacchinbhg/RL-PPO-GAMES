"""Render real evaluation images and decoded model outputs at one frame per decision."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from pixel_replay import host_path
out=project/"runs/pixels-only-01/official-wsl/controls/heldout-world-model"
with np.load(out/"alternative-futures.npz") as data:
    original, alternative = (data[k] for k in ("recorded","reversed"))
pred=[]
for name in ("recorded","reversed"):
    with np.load(out/(name+".npz")) as data:
        grid=data["frames"]
    red=np.all(grid[:,0,0]==[255,0,0],axis=-1)
    pred.append(grid[red].reshape(16,196,8,68,3).transpose(2,0,1,3,4)[:,:,78:118,4:64])
change=(np.abs(original.astype(np.int16)-alternative)>=8).any(-1).mean((1,2,3))
chosen=int(np.argsort(change, kind="stable")[(len(change)-1)//2])
ffmpeg=shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
ffprobe=shutil.which("ffprobe") or shutil.which("ffprobe.exe")
assert ffmpeg and ffprobe
temp=out/"button-order-test.partial.mp4"
font=ImageFont.load_default(size=22)
label=ImageFont.load_default(size=18)
encoder=subprocess.Popen([ffmpeg,"-hide_banner","-loglevel","error","-nostdin","-y",
    "-f","rawvideo","-pixel_format","rgb24","-video_size","1000x660","-framerate","16777216/6741504",
    "-i","pipe:0","-vf","fps=30","-c:v","libx264","-pix_fmt","yuv420p","-crf","16",
    "-movflags","+faststart",host_path(temp,ffmpeg)],stdin=subprocess.PIPE)
try:
    for k in range(16):
        canvas=Image.new("RGB",(1000,660),"#15181d")
        draw=ImageDraw.Draw(canvas)
        draw.text((20,14),f"Same starting screen, two button orders     |     Decision {k+1} of 16",
                  font=font,fill="#ffffff")
        for col,(title,real,model) in enumerate((("Recorded order",original,pred[0]),
                                                 ("Reversed order",alternative,pred[1]))):
            x=160+410*col
            draw.text((x,60),title,font=font,fill="#d2dcf0")
            for row,source in enumerate((real,model)):
                canvas.paste(Image.fromarray(source[chosen,k]).resize((390,260),Image.Resampling.NEAREST),
                             (x,98+280*row))
        draw.text((18,197),"Emerald ROM",font=label,fill="#ffffff")
        draw.text((18,477),"DreamerV3",font=label,fill="#ffffff")
        encoder.stdin.write(canvas.tobytes())
finally:
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Encoding failed")
meta=json.loads(subprocess.check_output([ffprobe,"-v","error","-show_entries",
    "stream=codec_name,width,height,nb_frames,duration","-of","json",host_path(temp,ffprobe)],text=True))
stream=meta["streams"][0]
assert stream["width"]==1000 and stream["height"]==660 and 6.3<float(stream["duration"])<6.6
temp.replace(out/"button-order-test.mp4")
meta.update(sequence=chosen,selection="lower median changed-pixel fraction across all eight windows",
            median_example_changed_pixel_fraction=float(change[chosen]))
(out/"clip-provenance.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta), flush=True)
