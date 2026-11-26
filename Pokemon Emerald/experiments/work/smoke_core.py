"""Development check for the native callbacks and state transport."""
from pathlib import Path
import sys
import numpy as np
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "outputs" / "EmeraldRL"))
import emerald

if len(sys.argv) > 1:
    rom = Path(sys.argv[1])
else:
    # Standalone ARM self-branch with a GBA header; only used to exercise the ABI.
    rom = root / "abi-test.gba"
    header = bytearray(1024)
    header[:4] = bytes.fromhex("feffffea")
    header[0xb2] = 0x96
    rom.write_bytes(header)
    emerald.verify_rom = lambda p: None
env = emerald.Emerald(root / "mgba_libretro.dll", rom)
for i in range(24):
    env.step(i % 9)
snap = env.save()
d = env.descriptor()
outcomes = [env.step(i) for i in range(9)]
env.load(snap)
assert d == env.descriptor()
for i, (image, desc) in enumerate(outcomes):
    image2, desc2 = env.step(i)
    assert desc2 == desc and np.array_equal(image, image2)
print("Verified 9-action replay from emulator state", len(snap), d, env.pixel_format, flush=True)
env.close()
