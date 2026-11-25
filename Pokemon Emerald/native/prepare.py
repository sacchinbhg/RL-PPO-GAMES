"""Install the trainer bridge into a disposable, pinned upstream checkout."""
import argparse
import json
from pathlib import Path
import re
import subprocess

PIN = "fd83f5b6e609b61b0b10777e32c74343e1a81b41"
HERE = Path(__file__).resolve().parent


def text_if_changed(path, value):
    if not path.exists() or path.read_text(encoding="utf-8") != value:
        path.write_text(value, encoding="utf-8", newline="\n")


def prepare(root, windows):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PIN:
        raise ValueError(f"Expected upstream {PIN}, found {head}")
    patches = (["windows-coff.patch"] if windows else []) + ["native-flash.patch", "native-wram.patch"]
    for name in patches:
        patch = str(HERE / name)
        if subprocess.run(["git", "apply", "--reverse", "--check", patch], cwd=root,
                          capture_output=True).returncode:
            subprocess.run(["git", "apply", patch], cwd=root, check=True)
    groups = json.loads((root / "data/maps/map_groups.json").read_text())
    counts = ", ".join(str(len(groups[g])) for g in groups["group_order"])
    text_if_changed(root / "include/rl_map_counts.h", f"static const int rlMapCounts[] = {{{counts}}};\n")
    # The asset generator's recursive make already has tools and generated maps.
    # Propagating SETUP_PREREQS=1 makes upstream's "make generated" call itself.
    assets = root / "tools/generate_wasm_assets.py"
    original = assets.read_text(encoding="utf-8")
    text_if_changed(assets, original.replace("'SETUP_PREREQS=1', target", "'SETUP_PREREQS=0', target"))
    makefile = root / "Makefile"
    text_if_changed(makefile, makefile.read_text(encoding="utf-8").replace(
        "$(MAKE) generated | sed", "$(MAKE) SETUP_PREREQS=0 generated | sed"))
    # A single upstream reset writes 256 MiB, including unmapped GBA addresses.
    # Keep never-written zero pages shared across our independent OS workers.
    engine = root / "tools/native/native_engine.c"
    code = engine.read_text(encoding="utf-8")
    clear = """/* EmeraldRL sparse reset: the image loader zeroes static storage at startup. */
    if (initialGameData) {
        static const unsigned char zeros[65536] = {0};
        for (size_t i = 0; i < sizeof(gNativeMemory); i += sizeof(zeros)) {
            if (memcmp(gNativeMemory + i, zeros, sizeof(zeros)) != 0)
                memset(gNativeMemory + i, 0, sizeof(zeros));
        }
    }"""
    if clear not in code:
        old = "memset(gNativeMemory, 0, sizeof(gNativeMemory));"
        if code.count(old) != 1:
            raise ValueError("Upstream reset implementation does not match the pinned build")
        engine.write_text(code.replace(old, clear), encoding="utf-8", newline="\n")
    menu = (root / "src/main_menu.c").read_text(encoding="utf-8")
    tasks = sorted(set(re.findall(r"^static void (Task_NewGameBirchSpeech\w*)\(u8\);", menu, re.M)))
    if not tasks:
        raise ValueError("Could not identify Birch introduction tasks")
    wrappers = {
        "title_screen.c": "int RlTitlePhase(void) { return gMain.callback2 == MainCB2 || gMain.callback2 == CB2_InitTitleScreen; }",
        "naming_screen.c": "int RlNamingPhase(void) { return gMain.callback2 == CB2_NamingScreen ? 4 : 0; }",
        "main_menu.c": """int RlMenuPhase(void) {
    if (gMain.callback2 != CB2_MainMenu) return 0;
    for (int i = 0; i < NUM_TASKS; ++i) {
        if (gTasks[i].isActive && (%s)) return 3;
    }
    return 2;
}""" % " || ".join("gTasks[i].func == " + task for task in tasks),
    }
    for name, wrapper in wrappers.items():
        source = root / "src" / name
        marker = "\n/* EmeraldRL read-only callback identity */"
        base = source.read_text(encoding="utf-8").partition(marker)[0]
        text_if_changed(source, base + marker + "\n#if NATIVE\n" + wrapper + "\n#endif\n")
    for source, dest in [("observe.c", "src/rl_observe.c"), ("bridge.c", "tools/native/rl_bridge.c"),
                         ("build_target.mk", "rl-build.mk")]:
        text_if_changed(root / dest, (HERE / source).read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout")
    parser.add_argument("--windows", action="store_true")
    args = parser.parse_args()
    prepare(args.checkout, args.windows)
