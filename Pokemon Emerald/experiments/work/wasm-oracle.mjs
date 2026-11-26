import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import vm from 'node:vm';
import {join} from 'node:path';
const root = 'work/upstream-emerald';
const out = 'work/native-probe/wasm-oracle';
mkdirSync(out,{recursive:true});
const source = readFileSync(join(root,'web/app.js'),'utf8');
const context = vm.createContext({WebAssembly, Uint8Array, Uint16Array, Uint32Array});
vm.runInContext(source.slice(source.indexOf('function copy(src'),source.indexOf('function writeKeys()')),context);
const module = await WebAssembly.compile(readFileSync(join(root,'build/wasm/pokeemerald.wasm')));
const inst= await WebAssembly.instantiate(module,context.importsFor(module));
const e=inst.exports, u8=new Uint8Array(e.memory.buffer), u16=new Uint16Array(e.memory.buffer);
Object.assign(context,{u8,u16,u32:new Uint32Array(e.memory.buffer)});
u8.fill(255,0x0e000000,0x0e020000);
u16[0x04000130>>1]=1023;
e.AgbMain();
const events=new Map(),names=['a','b','select','start','right','left','up','down','r','l'];
for (const line of readFileSync(join(root,'tools/wasm_replays/mudkip_starter.txt'),'utf8').split(/\r?\n/)) {
  const p=line.split('#')[0].trim().split(/\s+/);
  if(p.length<3)continue;
  const f=Number(p.shift());
  if(!events.has(f))events.set(f,[]);
  events.get(f).push(p);
}
let held=0;
for(let f=0;f<=Math.max(...events.keys());f++){
  for(const [type,b,v] of events.get(f)||[])if(type==='button'){
    const m=1<<names.indexOf(b);held=v==='on'?held|m:held&~m;
  }
  u16[0x04000130>>1]=1023^held;
  e.WasmRunFrame();
  for(const [type,name] of events.get(f)||[])if(type==='screenshot'){
    e.WasmRenderFrame();const start=e.WasmDisplayBuffer();
    writeFileSync(join(out,name+'.rgba'),u8.subarray(start,start+240*160*4));
    for (const [tag,addr,len] of [['vram',0x06000000,0x18000],['palette',0x05000000,1024],['io',0x04000000,0x58]]) {
      writeFileSync(join(out,name+'.'+tag),u8.subarray(addr,addr+len));
    }
  }
}
console.log('Reference frames and VRAM captured.');
