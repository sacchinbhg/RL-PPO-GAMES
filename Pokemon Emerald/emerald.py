"""Read-only Emerald observations and controller input through mGBA's libretro ABI."""

import ctypes as c
import hashlib
from functools import lru_cache
from pathlib import Path
import struct
import zlib

import numpy as np


ROM_SHA1 = "f3ae088181bf583e55daf962a92bb46f4f1d07b7"
BUTTONS = ("wait", "A", "B", "Start", "Up", "Down", "Left", "Right", "Select")
BUTTON_IDS = (None, 8, 0, 3, 4, 5, 6, 7, 2)
PHASES = ("boot", "title", "main-menu", "Birch", "naming", "overworld", "battle", "other")
REPEAT = 24


class GameInfo(c.Structure):
    _fields_ = [("path", c.c_char_p), ("data", c.c_void_p),
                ("size", c.c_size_t), ("meta", c.c_char_p)]


class Region(c.Structure):
    _fields_ = [("flags", c.c_uint64), ("ptr", c.c_void_p), ("offset", c.c_size_t),
                ("start", c.c_size_t), ("select", c.c_size_t), ("disconnect", c.c_size_t),
                ("length", c.c_size_t), ("addrspace", c.c_char_p)]


class MemoryMap(c.Structure):
    _fields_ = [("regions", c.POINTER(Region)), ("count", c.c_uint)]


def verify_rom(path):
    with Path(path).open("rb") as reader:
        checksum = hashlib.file_digest(reader, "sha1").hexdigest()
    if checksum != ROM_SHA1:
        raise ValueError(f"Expected English Emerald revision 0 ({ROM_SHA1}), got {checksum}")


class Emerald:
    """One instance per OS process. RAM addresses are for the verified English ROM."""

    def __init__(self, core, rom):
        verify_rom(rom)
        self.lib = c.CDLL(str(Path(core).resolve()))
        self.regions = {}
        self.callbacks = []
        self.buttons = 0
        self.pixel_format = 0
        self.capture = False
        self.screen = np.zeros((160, 240, 3), dtype=np.uint8)

        def register(name, fn, result, *args):
            cb = c.CFUNCTYPE(result, *args)(fn)
            self.callbacks.append(cb)
            setter = getattr(self.lib, "retro_set_" + name)
            setter.argtypes = [type(cb)]
            setter(cb)

        register("environment", self.environment, c.c_bool, c.c_uint, c.c_void_p)
        register("video_refresh", self.video, None, c.c_void_p, c.c_uint, c.c_uint, c.c_size_t)
        register("audio_sample_batch", lambda data, n: n, c.c_size_t, c.c_void_p, c.c_size_t)
        register("audio_sample", lambda l, r: None, None, c.c_int16, c.c_int16)
        register("input_poll", lambda: None, None)
        register("input_state", self.input, c.c_int16, *([c.c_uint] * 4))
        self.lib.retro_init()
        self.lib.retro_load_game.argtypes = [c.POINTER(GameInfo)]
        self.lib.retro_load_game.restype = c.c_bool
        data = Path(rom).read_bytes()
        self.rom_data = c.create_string_buffer(data)
        info = GameInfo(str(rom).encode(), c.addressof(self.rom_data), len(data), None)
        if not self.lib.retro_load_game(c.byref(info)):
            raise RuntimeError("mGBA could not load Emerald")
        self.lib.retro_serialize_size.restype = c.c_size_t
        self.lib.retro_get_memory_data.argtypes = [c.c_uint]
        self.lib.retro_get_memory_data.restype = c.c_void_p
        self.lib.retro_get_memory_size.argtypes = [c.c_uint]
        self.lib.retro_get_memory_size.restype = c.c_size_t
        for name in ("retro_serialize", "retro_unserialize"):
            func = getattr(self.lib, name)
            func.argtypes = [c.c_void_p, c.c_size_t]
            func.restype = c.c_bool
        self.capture = True
        self.lib.retro_run()
        if not {0x02000000, 0x03000000}.issubset(self.regions):
            raise RuntimeError("mGBA did not expose Emerald's working RAM")
        self.boot = self.save()
        self.boot_image = self.small_screen()

    def environment(self, cmd, data):
        # The experimental flag is part of the libretro memory-map/bitmask commands.
        cmd &= 0xFFFF
        if cmd == 10:
            self.pixel_format = c.c_int.from_address(data).value
            return self.pixel_format in (0, 1, 2)
        if cmd == 36:
            mmap = MemoryMap.from_address(data)
            for i in range(mmap.count):
                r = mmap.regions[i]
                if r.start in (0x02000000, 0x03000000) and r.ptr:
                    self.regions[r.start] = (r.ptr + r.offset, r.length)
            return True
        return cmd == 51  # input-mask support

    def input(self, port, device, index, button):
        if port != 0 or device != 1:
            return 0
        return self.buttons if button == 256 else (self.buttons >> button) & 1

    def video(self, ptr, width, height, pitch):
        if not self.capture or not ptr or (height, width) != (160, 240):
            return
        raw = c.string_at(ptr, height * pitch)
        dtype = np.uint32 if self.pixel_format == 1 else np.uint16
        p = np.frombuffer(raw, dtype=dtype).reshape(height, -1)[:, :width]
        if self.pixel_format == 1:
            components = (p >> 16, p >> 8, p)
        else:
            green_bits = 6 if self.pixel_format == 2 else 5
            components = (((p >> (5 + green_bits)) & 31) << 3,
                          ((p >> 5) & (2**green_bits - 1)) << (8 - green_bits),
                          (p & 31) << 3)
        self.screen = np.stack(components, axis=-1).astype(np.uint8)

    def small_screen(self):
        # RGB for the policy and the archived discovery gallery.
        return self.screen[::4, ::4].copy()

    def ram(self, addr, length):
        base = addr & 0xFF000000
        ptr, size = self.regions[base]
        offset = addr - base
        if not 0 <= offset <= size - length:
            raise ValueError(f"Out-of-range GBA RAM read {addr:#x}")
        return c.string_at(ptr + offset, length)

    def descriptor(self):
        callback = int.from_bytes(self.ram(0x030022C4, 4), "little") & ~1
        stage = {0x080AAB2C: 1, 0x080AA7A4: 1, 0x0802F6B0: 2,
                 0x080E4F58: 4, 0x08085E5C: 5, 0x08038420: 6}.get(callback, 0)
        if stage == 2:
            tasks = self.ram(0x03005E00, 16 * 40)
            for off in range(0, len(tasks), 40):
                func = struct.unpack_from("<I", tasks, off)[0] & ~1
                if tasks[off + 4] and 0x080307B0 <= func <= 0x08031630:
                    stage = 3
                    break
        sb = int.from_bytes(self.ram(0x03005D8C, 4), "little")
        if not (0x02025A00 <= sb <= 0x02025A7C and sb % 4 == 0):
            return (stage, -1, -1, 0, 0, 0, 0, 0, 0)
        intro = int.from_bytes(self.ram(sb + 0x139C + 2 * 0x92, 2), "little")
        # Until the moving-truck script has run, initialized RAM can look like map 0:0.
        if intro == 0:
            return (stage, -1, -1, 0, 0, 0, 0, 0, 0)
        x, y, group, num = struct.unpack("<hhBB", self.ram(sb, 6))
        flags = int.from_bytes(self.ram(sb + 0x1270 + 0x860 // 8, 2), "little")
        clock_set = (self.ram(sb + 0x1270 + 0x895 // 8, 1)[0] >> (0x895 % 8)) & 1
        if stage == 0:
            stage = 7
        # Both genders have identical credit for arriving in the truck.
        story = max(1, intro - 1)
        return (stage, group, num, x // 2, y // 2, story, clock_set,
                flags & 1, (flags >> 7) & 255 | ((flags >> 4) & 1) << 8)

    def step(self, action):
        button = BUTTON_IDS[action]
        for frame in range(REPEAT):
            self.buttons = (1 << button) if button is not None and frame < 14 else 0
            self.capture = frame == REPEAT - 1
            self.lib.retro_run()
        return self.small_screen(), self.descriptor()

    @lru_cache(maxsize=1024)
    def _header(self, group, number):
        def pointer(address):
            return struct.unpack_from("<I", self.rom_data, address - 0x08000000)[0]
        group_table = pointer(0x08486578 + 4 * group)
        if not 0x08000000 <= group_table + 4 * number < 0x09000000:
            return None
        header = pointer(group_table + 4 * number)
        if not 0x08000000 <= header <= 0x09000000 - 28:
            return None
        return (pointer(header + 4), struct.unpack_from("<H", self.rom_data, header - 0x08000000 + 18)[0])

    def verified_story(self, key):
        if key[0] not in (5, 6) or key[1] < 0 or key[5] > 6:
            return None
        read = lambda addr, size: int.from_bytes(self.ram(addr, size), "little")
        if self._header(key[1], key[2]) != (read(0x02037318 + 4, 4), read(0x02037318 + 18, 2)):
            return None
        sb = read(0x03005D8C, 4)
        var = lambda offset: read(sb + 0x139c + 2 * offset, 2)
        return bool(var(0x50) >= 1 and var(0x8d) >= 3), 1 <= read(0x020244e9, 1) <= 6

    def save(self):
        n = self.lib.retro_serialize_size()
        block = c.create_string_buffer(n)
        if not self.lib.retro_serialize(block, n):
            raise RuntimeError("mGBA serialization failed")
        # Libretro unserialize restores the RTC, but leaves battery-backed flash to
        # the frontend. Include it to keep save slots isolated across branches.
        flash_size = self.lib.retro_get_memory_size(0)
        flash = c.string_at(self.lib.retro_get_memory_data(0), flash_size)
        return zlib.compress(struct.pack("<I", n) + block.raw + flash, level=1)

    def load(self, state):
        raw = zlib.decompress(state)
        n = struct.unpack_from("<I", raw)[0]
        block, flash = raw[4:4+n], raw[4+n:]
        if not self.lib.retro_unserialize(block, n):
            raise RuntimeError("mGBA state restore failed")
        if len(flash) != self.lib.retro_get_memory_size(0):
            raise RuntimeError("mGBA save-memory geometry changed across restore")
        if flash:
            c.memmove(self.lib.retro_get_memory_data(0), flash, len(flash))
        self.buttons = 0

    def close(self):
        self.lib.retro_unload_game()
        self.lib.retro_deinit()
