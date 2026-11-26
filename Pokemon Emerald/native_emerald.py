"""Process-isolated, headless native Emerald. Archives are deterministic input replays."""
import ctypes as c
import hashlib
from pathlib import Path
import struct

import numpy as np

from emerald import BUTTONS, verify_rom

NATIVE_KEYS = (0, 1, 2, 8, 64, 128, 32, 16, 4)


class NativeEmerald:
    def __init__(self, core, rom):
        verify_rom(rom)
        core = Path(core).resolve()
        self.prefix = b"ERL-native-v1\0" + hashlib.sha256(core.read_bytes()).digest()
        self.lib = c.CDLL(str(core))
        for name, args, result in [
            ("rl_power_on", [], c.c_int), ("rl_reset", [], c.c_int), ("rl_close", [], None),
            ("rl_step", [c.c_uint8], None), ("rl_frames", [c.c_uint32, c.c_uint], None),
            ("rl_observe", [c.POINTER(c.c_int)], None), ("rl_screen", [], c.POINTER(c.c_uint8)),
            ("rl_render", [], c.POINTER(c.c_uint8)), ("rl_replay", [c.c_char_p, c.c_uint], c.c_int),
        ]:
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        if not self.lib.rl_reset():
            raise RuntimeError("Could not create the native Emerald engine")
        rgba = np.ctypeslib.as_array(self.lib.rl_screen(), shape=(160 * 240 * 4,))
        self.screen = rgba.reshape(160, 240, 4)[:, :, :3]
        self._values = (c.c_int * 13)()
        self.actions = bytearray()
        self.boot = self.save()
        self.boot_image = self.small_screen()

    def small_screen(self):
        return self.screen[::4, ::4].copy()

    def inspect(self):
        self.lib.rl_observe(self._values)
        return tuple(self._values)

    def descriptor(self):
        return self.inspect()[:9]

    def verified_story(self, key):
        if key[0] not in (5, 6) or key[1] < 0:
            return None
        return bool(self._values[9]), 1 <= self._values[11] <= 6

    def step(self, action):
        if not 0 <= action < len(BUTTONS):
            raise ValueError(f"Unknown action {action}")
        self.lib.rl_step(action)
        self.actions.append(action)
        return self.small_screen(), self.descriptor()

    def save(self):
        return self.prefix + struct.pack("<I", len(self.actions)) + self.actions

    def load(self, state, *, render_intermediate=True):
        if not state.startswith(self.prefix) or len(state) < len(self.prefix) + 4:
            raise ValueError("Archive belongs to a different engine or build")
        size, = struct.unpack_from("<I", state, len(self.prefix))
        actions = state[len(self.prefix) + 4:]
        if len(actions) != size:
            raise ValueError("Could not reproduce native Emerald archive state")
        if render_intermediate:
            if not self.lib.rl_replay(actions, size):
                raise ValueError("Could not reproduce native Emerald archive state")
        else:
            if any(b >= len(NATIVE_KEYS) for b in actions) or not self.lib.rl_reset():
                raise ValueError("Could not reproduce native Emerald archive state")
            for b in actions:
                self.lib.rl_frames(NATIVE_KEYS[b], 14)
                self.lib.rl_frames(0, 10)
            self.lib.rl_render()
        self.actions = bytearray(actions)

    def close(self):
        self.lib.rl_close()
