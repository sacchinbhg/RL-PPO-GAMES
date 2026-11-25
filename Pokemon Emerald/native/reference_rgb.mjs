// Generate the oracle hashes with Node and the pinned upstream browser engine.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import vm from 'node:vm';

const checkout = process.argv[2];
if (!checkout) throw new Error('Usage: node reference_rgb.mjs <upstream checkout with make wasm output>');
const app = readFileSync(join(checkout, 'web/app.js'), 'utf8');
const start = app.indexOf('function copy(src');
const end = app.indexOf('function writeKeys()');
if (start < 0 || end < start) throw new Error('Unexpected upstream BIOS interface');
const host = vm.createContext({ WebAssembly, Uint8Array, Uint16Array, Uint32Array });
vm.runInContext(app.slice(start, end), host);
const compiled = await WebAssembly.compile(readFileSync(join(checkout, 'build/wasm/pokeemerald.wasm')));
const { exports: game } = await WebAssembly.instantiate(compiled, host.importsFor(compiled));
const memory = game.memory.buffer;
const u8 = new Uint8Array(memory), u16 = new Uint16Array(memory);
Object.assign(host, { u8, u16, u32: new Uint32Array(memory) });
u8.fill(255, 0x0e000000, 0x0e020000);
u16[0x04000130 >> 1] = 1023;
game.AgbMain();

const names = ['a', 'b', 'select', 'start', 'right', 'left', 'up', 'down', 'r', 'l'];
const scenes = new Set(['after-clock', 'outside-rival-house', 'mudkip-battle']);
const events = new Map();
for (const line of readFileSync(join(checkout, 'tools/wasm_replays/mudkip_starter.txt'), 'utf8').split('\n')) {
  const [frame, ...args] = line.split('#')[0].trim().split(/\s+/);
  if (args.length < 2) continue;
  const index = Number(frame);
  if (!events.has(index)) events.set(index, []);
  events.get(index).push(args);
}
let held = 0;
const results = {};
const last = Math.max(...events.keys());
for (let frame = 0; frame <= last; frame++) {
  const entries = events.get(frame) ?? [];
  for (const [kind, name, state] of entries) {
    if (kind !== 'button') continue;
    const mask = 1 << names.indexOf(name);
    held = state === 'on' ? held | mask : held & ~mask;
  }
  u16[0x04000130 >> 1] = 1023 ^ held;
  game.WasmRunFrame();
  for (const [kind, name] of entries) {
    if (kind !== 'screenshot' || !scenes.has(name)) continue;
    game.WasmRenderFrame();
    const base = game.WasmDisplayBuffer();
    const rgb = u8.slice(base, base + 240 * 160 * 4).filter((_, i) => i % 4 !== 3);
    results[name] = createHash('sha256').update(rgb).digest('hex');
  }
}
console.log(JSON.stringify(results, null, 2));
