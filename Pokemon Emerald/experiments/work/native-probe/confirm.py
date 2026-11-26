from pathlib import Path
from collections import defaultdict
import ctypes as C
import os
import numpy as np
from PIL import Image,ImageDraw
root=Path.cwd(); guard=os.add_dll_directory(str(root/'work/msys64/ucrt64/bin')); dll=C.CDLL(str(root/'work/native-probe/fixed.dll')); dll.rl_render.restype=C.c_void_p
p=root/'work/upstream-emerald/tools/wasm_replays/mudkip_starter.txt'; events=defaultdict(list); names={}; masks=['a','b','select','start','right','left','up','down','r','l']
for ln in p.read_text().splitlines():
 f=ln.split();
 if len(f)<3 or not f[0].isdigit():continue
 t=int(f[0])
 if f[1]=='button':events[t].append((masks.index(f[2]),f[3]=='on'))
 if f[1]=='screenshot':names[t]=f[2]
for t in (200,400,700,1300,2000,3000):names[t]='frame'
held=0;last=0; out=(C.c_int*13)(); sheet=Image.new('RGB',(240*4,183*((len(names)+3)//4))); draw=ImageDraw.Draw(sheet); n=0;dll.rl_power_on()
for t in sorted(set(events)|set(names)):
 if last<t:dll.rl_frames(held,t-last)
 for bit,setting in events[t]:
  if setting:held|=1<<bit
  else:held&=~(1<<bit)
 dll.rl_frames(held,1);last=t+1
 if t in names:
  dll.rl_observe(out); a=np.frombuffer(C.string_at(dll.rl_render(),240*160*4),np.uint8).reshape(160,240,4);x=n%4*240;y=n//4*183;n+=1;sheet.paste(Image.fromarray(a[:,:,:3]),(x,y));draw.text((x+3,y+163),f'{t} {names[t]}');print(t,names[t],list(out))
sheet.save(root/'work/native-probe/actual.png')
